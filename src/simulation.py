"""Motore di calcolo Monte Carlo multi-asset con fiscalità progressiva, Totalizzazione UE (Italia-Germania) e TFR."""

from dataclasses import replace
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from config.settings import SimulationConfig
from src.pension import calcola_pensione_totale_ue
from src.taxes import calcola_stipendio_netto_automatico


def calcola_curva_spese_dinamica(
    config: SimulationConfig,
    inflazione: np.ndarray
) -> np.ndarray:
    """Costruisce la matrice delle spese lorde annue con eventuale effetto 'Retirement Smile'."""
    anni_totali = config.anni_vita
    num_sim = inflazione.shape[1]

    fattore_infl = np.ones((anni_totali, num_sim))
    fattore_infl[1:, :] = np.cumprod(1.0 + inflazione[1:anni_totali, :], axis=0)
    spese = config.prelievo_iniziale_annuo * fattore_infl

    if config.usa_curva_spese_smile:
        eta_anni = config.eta_iniziale + np.arange(anni_totali)[:, np.newaxis]
        fattore_smile = np.where(eta_anni < 55, 1.0, np.where(eta_anni < 75, 0.88, 1.05))
        spese *= fattore_smile

    return spese


def calcola_depositi_e_tfr(
    config: SimulationConfig,
    inflazione: np.ndarray,
    spese: np.ndarray,
    fattore_infl: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola in modo completamente vettorizzato i risparmi annui in accumulo applicando
    la fiscalità progressiva reale (Germania o Italia) e il TFR (solo Italia).
    """
    anni_acc = config.anni_acc
    num_sim = inflazione.shape[1]

    if fattore_infl is None:
        fattore_infl = np.cumprod(1.0 + inflazione, axis=0)

    ral_infl = np.empty((anni_acc, num_sim))
    ral_infl[0, :] = config.ral
    if anni_acc > 1:
        crescita = 1.0 + inflazione[1:anni_acc, :] + config.aumento_ral_annuo_reale
        ral_infl[1:, :] = config.ral * np.cumprod(crescita, axis=0)

    anni_col = np.arange(anni_acc)[:, np.newaxis]
    netto = calcola_stipendio_netto_automatico(
        ral=ral_infl,
        paese=config.paese,
        fattore_inflazione=fattore_infl[:anni_acc, :],
        tipo_sanita=config.tipo_sanita_germania,
        costo_mensile_pkv=config.costo_mensile_pkv_iniziale,
        anno=anni_col,
        inflazione_medica_extra=config.inflazione_medica_extra_annua
    )
    depositi = netto - spese[:anni_acc, :]

    tfr_maturato = np.zeros((anni_acc + 1, num_sim))
    if config.abilita_tfr and not config.is_germania:
        for t in range(anni_acc):
            accantonamento = ral_infl[t, :] * config.aliquota_accantonamento_tfr
            tasso_riv = 0.015 + 0.75 * inflazione[t, :]
            tfr_maturato[t + 1, :] = tfr_maturato[t, :] * (1.0 + tasso_riv) + accantonamento

    return depositi, ral_infl, tfr_maturato


def accumula_multi_asset_tax_efficient(
    config: SimulationConfig,
    depositi: np.ndarray,
    md_eq: np.ndarray,
    md_bond: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fase di accumulo multi-asset con Inflow Rebalancing e tassazione del capital gain
    sul ribilanciamento attivo per la Bond Tent (negli ultimi 5 anni pre-FIRE).
    """
    anni_acc = config.anni_acc
    num_sim = depositi.shape[1]
    target_eq = config.target_equity_accumulo
    target_bd = config.target_bond_accumulo
    fattore_costi = (1.0 - config.ter_etf) * (1.0 - config.bollo)
    min_eq_tent = max(0.20, config.target_equity_fire - 0.15)
    aliquota_eq = config.aliquota_capital_gain

    cap_eq = np.zeros((anni_acc + 1, num_sim))
    cap_bd = np.zeros((anni_acc + 1, num_sim))
    cap_tot = np.zeros((anni_acc + 1, num_sim))
    carico_eq = np.zeros((anni_acc + 1, num_sim))
    carico_bd = np.zeros((anni_acc + 1, num_sim))

    cap_eq[0, :] = config.capitale_iniziale * target_eq
    cap_bd[0, :] = config.capitale_iniziale * target_bd
    cap_tot[0, :] = config.capitale_iniziale
    carico_eq[0, :] = cap_eq[0, :]
    carico_bd[0, :] = cap_bd[0, :]

    for year in range(anni_acc):
        anni_mancanti = anni_acc - year
        if config.usa_bond_tent and anni_mancanti <= 5:
            cur_eq_acc = target_eq - (target_eq - min_eq_tent) * ((5 - anni_mancanti) / 5.0)
        else:
            cur_eq_acc = target_eq

        eq_post = cap_eq[year, :] * md_eq[year, :] * fattore_costi
        bd_post = cap_bd[year, :] * md_bond[year, :] * fattore_costi
        tot_post = eq_post + bd_post

        dep = depositi[year, :]
        patrimonio_futuro = tot_post + dep
        target_eq_valore = patrimonio_futuro * cur_eq_acc

        # 1. Inflow rebalancing: orienta prima i nuovi flussi di risparmio per ridurre le vendite
        deficit_equity = np.maximum(0.0, target_eq_valore - eq_post)
        quota_dep_eq = np.minimum(dep, deficit_equity)
        quota_dep_bd = dep - quota_dep_eq

        eq_curr = eq_post + quota_dep_eq
        bd_curr = bd_post + quota_dep_bd
        carico_eq_curr = carico_eq[year, :] + quota_dep_eq
        carico_bd_curr = carico_bd[year, :] + quota_dep_bd

        # 2. Se siamo nella Bond Tent (ultimi 5 anni pre-FIRE) e i depositi non bastano a scendere
        # alla quota azionaria target, si vendono azioni per acquistare bond pagando le imposte sul capital gain
        if config.usa_bond_tent and anni_mancanti <= 5:
            tot_curr = eq_curr + bd_curr
            delta_eq = np.maximum(0.0, eq_curr - cur_eq_acc * tot_curr)
            mask_reb = delta_eq > 1e-4
            if np.any(mask_reb):
                r_c = np.where(eq_curr > 0, carico_eq_curr / np.maximum(eq_curr, 1e-9), 1.0)
                g_eq = np.maximum(0.0, 1.0 - r_c)

                # Quantità da vendere tenendo conto dell'imposta sul capital gain
                s_eq = np.where(
                    mask_reb,
                    np.minimum(eq_curr, delta_eq / np.maximum(1e-9, 1.0 - cur_eq_acc * g_eq * aliquota_eq)),
                    0.0
                )
                gain_reb = s_eq * g_eq
                if config.is_germania:
                    tassa_reb = np.maximum(0.0, gain_reb - config.franchigia_capital_gain_annua) * aliquota_eq
                else:
                    tassa_reb = gain_reb * aliquota_eq

                netto_acq_bd = s_eq - tassa_reb
                eq_curr -= s_eq
                bd_curr += netto_acq_bd
                carico_eq_curr = np.maximum(0.0, carico_eq_curr - s_eq * np.minimum(1.0, r_c))
                carico_bd_curr += netto_acq_bd

        cap_eq[year + 1, :] = eq_curr
        cap_bd[year + 1, :] = bd_curr
        cap_tot[year + 1, :] = eq_curr + bd_curr
        carico_eq[year + 1, :] = carico_eq_curr
        carico_bd[year + 1, :] = carico_bd_curr

    return cap_tot, cap_eq, cap_bd, carico_eq, carico_bd


def simula_decumulo_fire_avanzato(
    anni_acc_dyn: int,
    cap_eq_start: np.ndarray,
    cap_bd_start: np.ndarray,
    carico_eq_start: np.ndarray,
    carico_bd_start: np.ndarray,
    spese_lorde: np.ndarray,
    pensione_dettaglio: Dict[str, Any],
    m_eq: np.ndarray,
    md_eq: np.ndarray,
    md_bond: np.ndarray,
    inflazione: np.ndarray,
    config: SimulationConfig,
    cum_infl: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decumulo FIRE con timeline continua allineata, basi di carico separate, pensione UE (GRV + INPS)
    e tassazione del capital gain sul ribilanciamento dinamico della Bond Tent.
    """
    anni_restanti = config.anni_vita - anni_acc_dyn
    num_sim = cap_eq_start.shape[0]
    is_germania = config.is_germania
    is_pkv = config.tipo_sanita_germania.upper() == "PKV"
    aliquota_eq = config.aliquota_capital_gain
    aliquota_bd = config.aliquota_capital_gain_bond
    franchigia_cg = config.franchigia_capital_gain_annua
    fattore_costi = (1.0 - config.ter_etf) * (1.0 - config.bollo)
    anni_buffer = config.anni_buffer
    usa_guardrails = config.usa_guardrails_spesa
    soglia_gw = config.soglia_guardrail_wd
    fattore_taglio_gw = 1.0 - config.taglio_guardrail_spese
    abilita_pens = config.abilita_pensione_pubblica

    if cum_infl is None:
        cum_infl = np.cumprod(1.0 + inflazione, axis=0)

    eq = np.zeros((anni_restanti, num_sim))
    bd = np.zeros((anni_restanti, num_sim))
    tot = np.zeros((anni_restanti, num_sim))
    buffer = np.zeros((anni_restanti, num_sim))
    prelievi_totali = np.zeros((anni_restanti, num_sim))

    carico_eq = carico_eq_start.copy()
    carico_bd = carico_bd_start.copy()

    # Zainetto fiscale rotante a 4 slot senza copie di memoria
    z0 = np.zeros(num_sim)
    z1 = np.zeros(num_sim)
    z2 = np.zeros(num_sim)
    z3 = np.zeros(num_sim)

    eq[0, :] = cap_eq_start
    bd[0, :] = cap_bd_start
    tot[0, :] = cap_eq_start + cap_bd_start

    idx_grv = max(0, pensione_dettaglio["eta_accesso_grv"] - config.eta_iniziale)
    idx_inps = max(0, pensione_dettaglio["eta_accesso_inps"] - config.eta_iniziale)
    p_grv_base = pensione_dettaglio["pensione_netta_grv"]
    p_inps_base = pensione_dettaglio["pensione_netta_inps"]
    zuschuss_drv_val = pensione_dettaglio.get("zuschuss_drv", z0)

    denom_grv = np.maximum(cum_infl[idx_grv - 1, :], 1e-9) if idx_grv > 0 else 1.0
    denom_inps = np.maximum(cum_infl[idx_inps - 1, :], 1e-9) if idx_inps > 0 else 1.0

    costo_pkv_annuo_base = config.costo_mensile_pkv_fire_nominale * 12.0
    costo_gkv_annuo_base = config.costo_mensile_gkv_fire_nominale * 12.0
    infl_medica_base = 1.0 + config.inflazione_medica_extra_annua

    for t in range(anni_restanti - 1):
        idx_vita = anni_acc_dyn + t
        spesa_anno = spese_lorde[idx_vita, :].copy()

        f_infl_vita = 1.0 if idx_vita == 0 else cum_infl[idx_vita - 1, :]
        f_riv_grv = 1.0 if idx_vita <= idx_grv else (f_infl_vita / denom_grv)

        # Sanità in Germania durante il Decumulo FIRE
        if is_germania:
            if is_pkv:
                premio_pkv_tot = costo_pkv_annuo_base * f_infl_vita * (infl_medica_base ** idx_vita)
                if idx_vita < idx_grv or not abilita_pens:
                    spesa_anno += premio_pkv_tot
                else:
                    zuschuss_effettivo = zuschuss_drv_val * f_riv_grv
                    spesa_anno += np.maximum(0.0, premio_pkv_tot - zuschuss_effettivo)
            elif idx_vita < idx_grv or not abilita_pens:
                spesa_anno += costo_gkv_annuo_base * f_infl_vita

        # Copertura Pensione Pubblica Totale UE (con perequazione automatica all'inflazione)
        if abilita_pens:
            if idx_vita >= idx_grv and idx_vita >= idx_inps:
                f_riv_inps = 1.0 if idx_vita <= idx_inps else (f_infl_vita / denom_inps)
                spesa_anno = np.maximum(0.0, spesa_anno - (p_grv_base * f_riv_grv + p_inps_base * f_riv_inps))
            elif idx_vita >= idx_grv:
                spesa_anno = np.maximum(0.0, spesa_anno - p_grv_base * f_riv_grv)
            elif idx_vita >= idx_inps:
                f_riv_inps = 1.0 if idx_vita <= idx_inps else (f_infl_vita / denom_inps)
                spesa_anno = np.maximum(0.0, spesa_anno - p_inps_base * f_riv_inps)

        patrimonio_tot_disp = tot[t, :] + buffer[t, :]

        # Guardrails (Guyton-Klinger light)
        if usa_guardrails:
            tasso_istantaneo = spesa_anno / np.maximum(patrimonio_tot_disp, 1e-9)
            mercato_orso = (m_eq[idx_vita, :] < m_eq[idx_vita - 1, :]) if idx_vita > 0 else False
            condizione_taglio = (tasso_istantaneo > soglia_gw) | (mercato_orso & (tasso_istantaneo > 0.04))
            spesa_anno = np.where(condizione_taglio, spesa_anno * fattore_taglio_gw, spesa_anno)

        # Cash Buffer dinamico anticrisi
        mercato_up = (m_eq[idx_vita, :] > m_eq[idx_vita - 1, :]) if idx_vita > 0 else True
        target_buffer = np.where(mercato_up, spesa_anno * anni_buffer, spesa_anno)
        new_buffer = np.minimum(patrimonio_tot_disp, target_buffer)
        fabbisogno_da_portafoglio = np.maximum(0.0, new_buffer - buffer[t, :])

        # Outflow Rebalancing verso target_equity_fire (con supporto per Bond Tent Glidepath)
        cur_target_eq, _ = config.get_allocazione_anno(idx_vita, anni_acc_dyn)
        cur_target_bd = 1.0 - cur_target_eq
        peso_eq_attuale = eq[t, :] / np.maximum(tot[t, :], 1e-9)

        prelievo_eq = np.where(
            peso_eq_attuale >= cur_target_eq,
            np.minimum(eq[t, :], fabbisogno_da_portafoglio),
            np.maximum(0.0, fabbisogno_da_portafoglio - bd[t, :])
        )
        prelievo_bd = fabbisogno_da_portafoglio - prelievo_eq

        # Fiscalità separata e Minusvalenze sul prelievo
        rapporto_carico_eq = np.where(eq[t, :] > 0, carico_eq / np.maximum(eq[t, :], 1e-9), 1.0)
        rapporto_carico_bd = np.where(bd[t, :] > 0, carico_bd / np.maximum(bd[t, :], 1e-9), 1.0)

        gain_eq = prelievo_eq * np.maximum(0.0, 1.0 - rapporto_carico_eq)
        loss_eq = prelievo_eq * np.maximum(0.0, rapporto_carico_eq - 1.0)

        gain_bd = prelievo_bd * np.maximum(0.0, 1.0 - rapporto_carico_bd)
        loss_bd = prelievo_bd * np.maximum(0.0, rapporto_carico_bd - 1.0)

        nuove_minus = loss_eq + loss_bd
        minus_disponibili = z0 + z1 + z2 + z3

        if is_germania:
            gain_tot = gain_eq + gain_bd
            minus_usate = np.minimum(gain_tot, minus_disponibili)
            minus_disponibili = np.maximum(0.0, minus_disponibili - gain_tot)
            gain_post_minus = gain_tot - minus_usate
            tassa = np.maximum(0.0, gain_post_minus - franchigia_cg) * aliquota_eq
        else:
            minus_usate_eq = np.minimum(gain_eq, minus_disponibili)
            minus_disponibili = np.maximum(0.0, minus_disponibili - gain_eq)
            gain_eq_post = gain_eq - minus_usate_eq

            minus_usate_bd = np.minimum(gain_bd, minus_disponibili)
            minus_disponibili = np.maximum(0.0, minus_disponibili - gain_bd)
            gain_bd_post = gain_bd - minus_usate_bd

            tassa = (gain_eq_post * aliquota_eq) + (gain_bd_post * aliquota_bd)

        carico_eq = np.maximum(0.0, carico_eq - prelievo_eq * np.minimum(1.0, rapporto_carico_eq))
        carico_bd = np.maximum(0.0, carico_bd - prelievo_bd * np.minimum(1.0, rapporto_carico_bd))

        eq[t, :] = np.maximum(0.0, eq[t, :] - prelievo_eq)
        bd[t, :] = np.maximum(0.0, bd[t, :] - prelievo_bd)
        tot[t, :] = eq[t, :] + bd[t, :]

        # Ribilanciamento attivo per la Bond Tent con tassazione del capital gain
        if config.usa_bond_tent:
            tot_post_prel = tot[t, :]
            delta_eq = eq[t, :] - cur_target_eq * tot_post_prel
            mask_sell_eq = (delta_eq > 1e-4) & (tot_post_prel > 0)
            if np.any(mask_sell_eq):
                r_c_eq = np.where(eq[t, :] > 0, carico_eq / np.maximum(eq[t, :], 1e-9), 1.0)
                g_eq = np.maximum(0.0, 1.0 - r_c_eq)
                loss_reb_eq = np.where(r_c_eq > 1.0, r_c_eq - 1.0, 0.0)

                s_eq = np.where(
                    mask_sell_eq,
                    np.minimum(eq[t, :], delta_eq / np.maximum(1e-9, 1.0 - cur_target_eq * g_eq * aliquota_eq)),
                    0.0
                )
                gain_reb_eq = s_eq * g_eq
                nuove_minus += s_eq * loss_reb_eq

                minus_usate_reb = np.minimum(gain_reb_eq, minus_disponibili)
                minus_disponibili = np.maximum(0.0, minus_disponibili - gain_reb_eq)
                gain_post_m = gain_reb_eq - minus_usate_reb
                tassa_reb_eq = gain_post_m * aliquota_eq

                netto_acq_bd = s_eq - tassa_reb_eq
                eq[t, :] -= s_eq
                bd[t, :] += netto_acq_bd
                carico_eq = np.maximum(0.0, carico_eq - s_eq * np.minimum(1.0, r_c_eq))
                carico_bd += netto_acq_bd
                tot[t, :] = eq[t, :] + bd[t, :]

            delta_bd = bd[t, :] - cur_target_bd * tot[t, :]
            mask_sell_bd = (delta_bd > 1e-4) & (tot[t, :] > 0)
            if np.any(mask_sell_bd):
                r_c_bd = np.where(bd[t, :] > 0, carico_bd / np.maximum(bd[t, :], 1e-9), 1.0)
                g_bd = np.maximum(0.0, 1.0 - r_c_bd)
                loss_reb_bd = np.where(r_c_bd > 1.0, r_c_bd - 1.0, 0.0)

                s_bd = np.where(
                    mask_sell_bd,
                    np.minimum(bd[t, :], delta_bd / np.maximum(1e-9, 1.0 - cur_target_bd * g_bd * aliquota_bd)),
                    0.0
                )
                gain_reb_bd = s_bd * g_bd
                nuove_minus += s_bd * loss_reb_bd

                minus_usate_reb_bd = np.minimum(gain_reb_bd, minus_disponibili)
                minus_disponibili = np.maximum(0.0, minus_disponibili - gain_reb_bd)
                gain_post_m_bd = gain_reb_bd - minus_usate_reb_bd
                tassa_reb_bd = gain_post_m_bd * aliquota_bd

                netto_acq_eq = s_bd - tassa_reb_bd
                bd[t, :] -= s_bd
                eq[t, :] += netto_acq_eq
                carico_bd = np.maximum(0.0, carico_bd - s_bd * np.minimum(1.0, r_c_bd))
                carico_eq += netto_acq_eq
                tot[t, :] = eq[t, :] + bd[t, :]

        z3, z2, z1, z0 = z2, z1, z0, nuove_minus

        tot_prelevato = prelievo_eq + prelievo_bd
        spesa_effettiva = np.minimum(tot[t, :] + buffer[t, :], spesa_anno)
        buffer[t, :] = buffer[t, :] + tot_prelevato - tassa - spesa_effettiva

        next_idx = min(idx_vita + 1, config.anni_vita)
        eq[t + 1, :] = eq[t, :] * md_eq[next_idx, :] * fattore_costi
        bd[t + 1, :] = bd[t, :] * md_bond[next_idx, :] * fattore_costi
        tot[t + 1, :] = eq[t + 1, :] + bd[t + 1, :]
        buffer[t + 1, :] = buffer[t, :] * (1.0 + inflazione[next_idx, :])

        prelievi_totali[t, :] = tot_prelevato

    return tot + buffer, prelievi_totali


def esegui_simulazione_completa(
    config: SimulationConfig,
    m_eq: np.ndarray,
    md_eq: np.ndarray,
    md_bond: np.ndarray,
    inflazione: np.ndarray
) -> Dict[str, Any]:
    """Esegue la simulazione Monte Carlo completa (Accumulo + Decumulo per ogni età di ritiro + Metriche SWR)."""
    anni_acc = config.anni_acc
    anni_vita = config.anni_vita
    num_sim = m_eq.shape[1]

    cum_infl = np.cumprod(1.0 + inflazione, axis=0)
    spese = calcola_curva_spese_dinamica(config, inflazione)
    depositi, ral_infl, tfr_maturato = calcola_depositi_e_tfr(config, inflazione, spese, fattore_infl=cum_infl)
    cap_tot_acc, cap_eq_acc, cap_bd_acc, carico_eq_acc, carico_bd_acc = accumula_multi_asset_tax_efficient(
        config, depositi, md_eq, md_bond
    )

    capitale_vita = []
    pensioni_list = []
    pensioni_dettaglio_list = []

    percentuali_positivi = np.full((anni_acc, anni_vita), np.nan)
    percentuali_wd_rate_roll = np.full((anni_acc, anni_vita), np.nan)

    for dyn in range(anni_acc):
        if config.abilita_pensione_pubblica:
            p_det = calcola_pensione_totale_ue(
                paese_attuale=config.paese,
                anni_lavoro_futuri=dyn,
                anni_pregressi_italia=config.anni_pregressi_italia,
                anni_pregressi_germania=config.anni_pregressi_germania,
                ral_pregressa_italia=config.ral_pregressa_italia,
                ral_infl_serie=ral_infl,
                inflazione=inflazione,
                eta_iniziale=config.eta_iniziale,
                eta_pensione_target=config.eta_pensione,
                pil_reale=config.pil_reale_medio,
                tipo_sanita=config.tipo_sanita_germania,
                fattore_infl=cum_infl
            )
        else:
            zeros = np.zeros(num_sim)
            p_det = {
                "pensione_netta_grv": zeros,
                "pensione_lorda_grv": zeros,
                "zuschuss_drv": zeros,
                "eta_accesso_grv": config.eta_pensione,
                "pensione_netta_inps": zeros,
                "eta_accesso_inps": config.eta_pensione,
                "anni_ue_totali": 0,
                "anni_de_totali": 0,
                "anni_it_totali": 0
            }

        pensioni_dettaglio_list.append(p_det)
        pensioni_list.append(p_det["pensione_netta_grv"] + p_det["pensione_netta_inps"])

        cap_eq_start = cap_eq_acc[dyn, :].copy()
        cap_bd_start = cap_bd_acc[dyn, :].copy()
        carico_eq_start = carico_eq_acc[dyn, :].copy()
        carico_bd_start = carico_bd_acc[dyn, :].copy()

        if config.abilita_tfr and dyn > 0:
            liquidazione_tfr_netta = tfr_maturato[dyn, :] * (1.0 - config.tassazione_separata_tfr)
            quota_tfr_eq = liquidazione_tfr_netta * config.target_equity_accumulo
            quota_tfr_bd = liquidazione_tfr_netta * config.target_bond_accumulo
            cap_eq_start += quota_tfr_eq
            cap_bd_start += quota_tfr_bd
            carico_eq_start += quota_tfr_eq
            carico_bd_start += quota_tfr_bd

        cap_patrimonio_fire, prel_tot = simula_decumulo_fire_avanzato(
            dyn,
            cap_eq_start,
            cap_bd_start,
            carico_eq_start,
            carico_bd_start,
            spese,
            p_det,
            m_eq,
            md_eq,
            md_bond,
            inflazione,
            config,
            cum_infl=cum_infl
        )

        parte_acc = cap_tot_acc[:dyn, :]
        matrice_unita = np.concatenate((parte_acc, cap_patrimonio_fire), axis=0)[:anni_vita, :]
        capitale_vita.append(matrice_unita)

        # Calcolo vettorizzato 2D delle metriche di successo e SWR sostenibile
        cap_sub = matrice_unita[dyn:anni_vita, :]
        prel_sub = prel_tot[:anni_vita - dyn, :]
        is_attivo = cap_sub > 0

        percentuali_positivi[dyn, dyn:anni_vita] = np.mean(is_attivo, axis=1) * 100.0

        wr_matrice = np.zeros((anni_vita, num_sim))
        wr_matrice[dyn:anni_vita, :] = np.where(is_attivo, prel_sub / np.maximum(cap_sub, 1e-9), 10.0)
        wr_filtrata = uniform_filter1d(wr_matrice, size=3, axis=0, mode="nearest")

        percentuali_wd_rate_roll[dyn, dyn:anni_vita] = (
            np.mean(is_attivo & (wr_filtrata[dyn:anni_vita, :] < config.safe_wd_rate), axis=1) * 100.0
        )

    return {
        "depositi": depositi,
        "ral_infl": ral_infl,
        "spese": spese,
        "capitale_acc": cap_tot_acc,
        "capitale_vita": capitale_vita,
        "pensioni_list": pensioni_list,
        "pensioni_dettaglio_list": pensioni_dettaglio_list,
        "percentuali_positivi": percentuali_positivi,
        "percentuali_wd_rate_roll": percentuali_wd_rate_roll,
    }


def simula_stress_test_storico(
    config: SimulationConfig,
    anno_inizio_crisi: int,
    anni_accumulo: int
) -> np.ndarray:
    """
    Simula una singola traiettoria deterministica in cui al momento dell'accesso al FIRE
    scoppia la crisi storica dell'anno specificato (es. 1973, 2000, 2008, 2022).
    Ritorna il vettore del capitale di dimensione (config.anni_vita,).
    """
    from src.market_data import carica_dataset_sincrono_eur

    df = carica_dataset_sincrono_eur(config)
    durata = config.anni_vita + 1

    is_world = config.nome_dataset_azionario.lower() in ("msci_world", "world", "urth")
    col_stock = "msci_world_eur" if is_world else "sp500_eur"
    col_infl = "inflation_germany" if config.is_germania else "inflation_italy"

    anni_dataset = df["Year"].to_numpy()
    matches = np.where(anni_dataset == anno_inizio_crisi)[0]
    idx_crisi = int(matches[0]) if len(matches) > 0 else 0

    # L'anno di scoppio della crisi deve colpire il primo anno effettivo di decumulo FIRE
    # (ovvero il passaggio tra l'età di ritiro t=anni_accumulo e t+1=anni_accumulo+1)
    indici = (idx_crisi + np.arange(durata) - (anni_accumulo + 1)) % len(df)

    md_eq = (1.0 + df[col_stock].to_numpy()[indici])[:, np.newaxis]
    md_bd = (1.0 + df["bund_10y_tr_eur"].to_numpy()[indici])[:, np.newaxis]
    infl = df[col_infl].to_numpy()[indici][:, np.newaxis]
    m_eq = np.ones((durata, 1))
    m_eq[1:, :] = np.cumprod(md_eq[1:, :], axis=0)

    cfg_1 = replace(config, num_simulazioni=1)
    cum_infl = np.cumprod(1.0 + infl, axis=0)
    spese = calcola_curva_spese_dinamica(cfg_1, infl)
    depositi, ral_infl, tfr_maturato = calcola_depositi_e_tfr(cfg_1, infl, spese, fattore_infl=cum_infl)
    cap_tot_acc, cap_eq_acc, cap_bd_acc, carico_eq_acc, carico_bd_acc = accumula_multi_asset_tax_efficient(
        cfg_1, depositi, md_eq, md_bd
    )

    if cfg_1.abilita_pensione_pubblica:
        p_det = calcola_pensione_totale_ue(
            paese_attuale=cfg_1.paese,
            anni_lavoro_futuri=anni_accumulo,
            anni_pregressi_italia=cfg_1.anni_pregressi_italia,
            anni_pregressi_germania=cfg_1.anni_pregressi_germania,
            ral_pregressa_italia=cfg_1.ral_pregressa_italia,
            ral_infl_serie=ral_infl,
            inflazione=infl,
            eta_iniziale=cfg_1.eta_iniziale,
            eta_pensione_target=cfg_1.eta_pensione,
            pil_reale=cfg_1.pil_reale_medio,
            tipo_sanita=cfg_1.tipo_sanita_germania,
            fattore_infl=cum_infl
        )
    else:
        zeros = np.zeros(1)
        p_det = {
            "pensione_netta_grv": zeros,
            "pensione_lorda_grv": zeros,
            "zuschuss_drv": zeros,
            "eta_accesso_grv": cfg_1.eta_pensione,
            "pensione_netta_inps": zeros,
            "eta_accesso_inps": cfg_1.eta_pensione,
            "anni_ue_totali": 0,
            "anni_de_totali": 0,
            "anni_it_totali": 0
        }

    cap_eq_start = cap_eq_acc[anni_accumulo, :].copy()
    cap_bd_start = cap_bd_acc[anni_accumulo, :].copy()
    carico_eq_start = carico_eq_acc[anni_accumulo, :].copy()
    carico_bd_start = carico_bd_acc[anni_accumulo, :].copy()

    if cfg_1.abilita_tfr and anni_accumulo > 0:
        liq_tfr = tfr_maturato[anni_accumulo, :] * (1.0 - cfg_1.tassazione_separata_tfr)
        cap_eq_start += liq_tfr * cfg_1.target_equity_accumulo
        cap_bd_start += liq_tfr * cfg_1.target_bond_accumulo
        carico_eq_start += liq_tfr * cfg_1.target_equity_accumulo
        carico_bd_start += liq_tfr * cfg_1.target_bond_accumulo

    cap_fire, _ = simula_decumulo_fire_avanzato(
        anni_accumulo, cap_eq_start, cap_bd_start, carico_eq_start, carico_bd_start,
        spese, p_det, m_eq, md_eq, md_bd, infl, cfg_1, cum_infl=cum_infl
    )

    parte_acc = cap_tot_acc[:anni_accumulo, :]
    matrice_unita = np.concatenate((parte_acc, cap_fire), axis=0)
    return matrice_unita[:config.anni_vita, 0]


def esegui_analisi_sensitivita_strategie(
    config: SimulationConfig,
    m_eq: np.ndarray,
    md_eq: np.ndarray,
    md_bond: np.ndarray,
    inflazione: np.ndarray,
    eta_fire_rif: int = 45,
    num_sim_sens: int = 1500
) -> pd.DataFrame:
    """
    Esegue una scansione sistematica ottimizzata dell'albero dei parametri strategici a partire dalla configurazione utente:
    - Varia l'asset allocation Pre-FIRE (Accumulo) e Post-FIRE (Decumulo).
    - Varia la presenza o assenza della Bond Tent.
    - Varia il regime sanitario (GKV vs PKV se in Germania).
    Mantiene intatti tutti gli altri parametri inseriti dall'utente (RAL, spese, capitale, pensione, inflazione, ecc.).
    Ritorna un DataFrame ordinato per età minima di FIRE al 99% e tenuta al 1° percentile peggiore.
    """
    n_s = min(num_sim_sens, m_eq.shape[1])
    m_sub = m_eq[:, :n_s]
    md_sub = md_eq[:, :n_s]
    bd_sub = md_bond[:, :n_s]
    inf_sub = inflazione[:, :n_s]
    cum_infl = np.cumprod(1.0 + inf_sub, axis=0)

    eq_acc_list = [0.90, 0.80, 0.70, 0.60, 0.50]
    eq_fire_list = [0.70, 0.60, 0.50, 0.40, 0.30]
    sanita_opts = ["GKV", "PKV"] if config.is_germania else [config.tipo_sanita_germania]
    bond_tent_opts = [True, False]

    coppie_alloc = [(acc, fire) for acc in eq_acc_list for fire in eq_fire_list if fire <= acc]
    anno_rif = max(0, min(eta_fire_rif - config.eta_iniziale, config.anni_acc - 1))
    anni_vita = config.anni_vita

    # Pre-calcolo vettorizzato condiviso per regime sanitario per evitare ricalcoli inutili
    cache_sanita = {}
    for san in sanita_opts:
        c_s = replace(config, num_simulazioni=n_s, tipo_sanita_germania=san)
        spese_s = calcola_curva_spese_dinamica(c_s, inf_sub)
        depositi_s, ral_infl_s, tfr_s = calcola_depositi_e_tfr(c_s, inf_sub, spese_s, cum_infl)
        p_dets_s = [
            calcola_pensione_totale_ue(
                c_s.paese, d, c_s.anni_pregressi_italia, c_s.anni_pregressi_germania,
                c_s.ral_pregressa_italia, ral_infl_s, inf_sub, c_s.eta_iniziale,
                c_s.eta_pensione, c_s.pil_reale_medio, c_s.tipo_sanita_germania, cum_infl
            )
            for d in range(c_s.anni_acc)
        ]
        acc_map_s = {}
        for eq_a in eq_acc_list:
            c_a = replace(c_s, target_equity_accumulo=eq_a, target_bond_accumulo=1.0 - eq_a)
            acc_map_s[eq_a] = accumula_multi_asset_tax_efficient(c_a, depositi_s, md_sub, bd_sub)

        cache_sanita[san] = {
            "spese": spese_s,
            "tfr": tfr_s,
            "p_dets": p_dets_s,
            "acc_map": acc_map_s
        }

    records = []
    for sanita in sanita_opts:
        c_data = cache_sanita[sanita]
        spese_cur = c_data["spese"]
        p_dets_cur = c_data["p_dets"]
        tfr_cur = c_data["tfr"]

        for bt in bond_tent_opts:
            for (acc_eq, fire_eq) in coppie_alloc:
                cfg_run = replace(
                    config,
                    num_simulazioni=n_s,
                    tipo_sanita_germania=sanita,
                    usa_bond_tent=bt,
                    target_equity_accumulo=acc_eq,
                    target_bond_accumulo=1.0 - acc_eq,
                    target_equity_fire=fire_eq,
                    target_bond_fire=1.0 - fire_eq
                )
                cap_tot_acc, cap_eq_acc, cap_bd_acc, carico_eq_acc, carico_bd_acc = c_data["acc_map"][acc_eq]

                def eval_dyn(dyn: int):
                    eq_start = cap_eq_acc[dyn, :].copy()
                    bd_start = cap_bd_acc[dyn, :].copy()
                    ceq_start = carico_eq_acc[dyn, :].copy()
                    cbd_start = carico_bd_acc[dyn, :].copy()
                    if cfg_run.abilita_tfr and dyn > 0:
                        liq_tfr = tfr_cur[dyn, :] * (1.0 - cfg_run.tassazione_separata_tfr)
                        eq_start += liq_tfr * acc_eq
                        bd_start += liq_tfr * (1.0 - acc_eq)
                        ceq_start += liq_tfr * acc_eq
                        cbd_start += liq_tfr * (1.0 - acc_eq)

                    cap_fire, _ = simula_decumulo_fire_avanzato(
                        dyn, eq_start, bd_start, ceq_start, cbd_start,
                        spese_cur, p_dets_cur[dyn], m_sub, md_sub, bd_sub, inf_sub,
                        cfg_run, cum_infl=cum_infl
                    )
                    idx_end = anni_vita - dyn - 1
                    succ = float(np.mean(cap_fire[idx_end, :] > 0) * 100.0)
                    return succ, cap_fire

                succ_cache = {}
                def get_succ(d: int):
                    if d not in succ_cache:
                        succ, c_f = eval_dyn(d)
                        succ_cache[d] = (succ, c_f)
                    return succ_cache[d]

                # Valutazione all'anno di riferimento (es. 45a o selezione utente)
                succ_rif, c_f_rif = get_succ(anno_rif)
                p01_rif = float(np.percentile(c_f_rif[-1, :], 1))
                p50_rif = float(np.median(c_f_rif[-1, :]))

                # Ricerca binaria efficiente per la prima età con successo >= 99%
                low, high = 0, config.anni_acc - 1
                min_dyn_99 = 999
                while low <= high:
                    mid = (low + high) // 2
                    s_mid, _ = get_succ(mid)
                    if s_mid >= 99.0:
                        min_dyn_99 = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                eta_fire_99 = int(config.eta_iniziale + min_dyn_99) if min_dyn_99 < 900 else 999

                # Ricerca binaria per la prima età con successo >= 95%
                low, high = 0, min(min_dyn_99, config.anni_acc - 1)
                min_dyn_95 = 999
                while low <= high:
                    mid = (low + high) // 2
                    s_mid, _ = get_succ(mid)
                    if s_mid >= 95.0:
                        min_dyn_95 = mid
                        high = mid - 1
                    else:
                        low = mid + 1
                eta_fire_95 = int(config.eta_iniziale + min_dyn_95) if min_dyn_95 < 900 else 999

                s_73 = simula_stress_test_storico(cfg_run, 1973, anno_rif)[-1]
                s_00 = simula_stress_test_storico(cfg_run, 2000, anno_rif)[-1]
                s_08 = simula_stress_test_storico(cfg_run, 2008, anno_rif)[-1]
                s_22 = simula_stress_test_storico(cfg_run, 2022, anno_rif)[-1]
                n_crisi_ok = sum([s_73 > 0, s_00 > 0, s_08 > 0, s_22 > 0])
                peggior_crisi = min(s_73, s_00, s_08, s_22)

                records.append({
                    "Sanità": sanita,
                    "Bond Tent": "Sì" if bt else "No",
                    "Azioni Pre-FIRE": f"{int(acc_eq * 100)}%",
                    "Bond Pre-FIRE": f"{int((1 - acc_eq) * 100)}%",
                    "Azioni Post-FIRE": f"{int(fire_eq * 100)}%",
                    "Bond Post-FIRE": f"{int((1 - fire_eq) * 100)}%",
                    "Età Min 99%": eta_fire_99,
                    "Età Min 95%": eta_fire_95,
                    f"Succ a {eta_fire_rif}a (%)": round(succ_rif, 1),
                    f"1° Perc 100a (Ritiro {eta_fire_rif}a)": round(p01_rif),
                    f"Mediana 100a (Ritiro {eta_fire_rif}a)": round(p50_rif),
                    "Crisi Storiche OK / 4": n_crisi_ok,
                    "Peggiore Shock Storico (€)": round(peggior_crisi),
                    "_acc_val": acc_eq,
                    "_fire_val": fire_eq,
                    "_bt_val": bt,
                    "_sanita_val": sanita
                })

    df = pd.DataFrame(records)
    col_succ = f"Succ a {eta_fire_rif}a (%)"
    col_p01 = f"1° Perc 100a (Ritiro {eta_fire_rif}a)"
    df = df.sort_values(by=["Età Min 99%", col_succ, col_p01], ascending=[True, False, False]).reset_index(drop=True)
    return df

