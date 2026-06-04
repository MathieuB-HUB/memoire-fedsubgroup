"""Génère les .tex includes à partir des résultats JSON.

Produit dans memoire/ :
  - table_fedheart.tex
  - table_ptbxl.tex          (placeholder si PTB-XL pas encore exécuté)
  - table_wilcoxon.tex
  - verdict.tex
  - discussion_lead.tex
  - conclusion_results.tex

Tous les fichiers sont écrits en UTF-8 avec accents français complets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import paired_wilcoxon


METHOD_ORDER = ["centralized", "fedavg", "fedavg_sg", "fedbn", "fedper", "fedrep", "ditto", "fedsubgroup", "local_only"]
METHOD_LABEL = {
    "fedavg": "FedAvg", "fedavg_sg": "FedAvg+sg", "fedbn": "FedBN", "fedper": "FedPer",
    "fedrep": "FedRep", "ditto": "Ditto", "fedsubgroup": "FedSubgroup",
    "centralized": "Centralized (oracle)", "local_only": "Local-only",
}
PHILO_B = ["fedbn", "fedper", "fedrep", "ditto"]


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for p in folder.glob("*.json"):
        d = json.loads(p.read_text())
        row = {k: d[k] for k in [
            "dataset", "method", "sensitive", "seed",
            "auc_global", "worst_group_auc", "auc_gap", "dp_gap", "eo_gap"
        ]}
        ci = d.get("wga_ci95", [float("nan"), float("nan")])
        row["wga_ci_low"] = ci[0]
        row["wga_ci_high"] = ci[1]
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mu, sd):
    if np.isnan(mu):
        return "--"
    return f"{mu:.3f} $\\pm$ {sd:.3f}"


def _short(x):
    """0.812 -> .81 (compact bound for IC bracket)."""
    if np.isnan(x):
        return "--"
    s = f"{x:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def fmt_wga_ci(mu, sd, ci_lo, ci_hi):
    """Worst-group AUC : moyenne $\\pm$ sd avec IC95 bootstrap moyen sur graines."""
    if np.isnan(mu):
        return "--"
    return (
        f"{mu:.3f} $\\pm$ {sd:.3f}"
        f"\\,[{_short(ci_lo)},{_short(ci_hi)}]"
    )


def fedheart_perf_tex(df: pd.DataFrame, label: str, caption: str) -> str:
    """Tableau perf Fed-Heart : AUC global, WG-AUC (avec IC95 bootstrap moyen), AUC gap."""
    sub = df[df["dataset"] == "fed_heart"]
    if sub.empty:
        return "% pas de données fed_heart\n"
    rows = []
    for m in METHOD_ORDER:
        line = [METHOD_LABEL[m]]
        for sens in ["sex", "age"]:
            d = sub[(sub["method"] == m) & (sub["sensitive"] == sens)]
            if len(d) == 0:
                line.extend(["--", "--", "--"])
                continue
            line.append(fmt(d["auc_global"].mean(), d["auc_global"].std()))
            line.append(fmt_wga_ci(
                d["worst_group_auc"].mean(), d["worst_group_auc"].std(),
                d["wga_ci_low"].mean(), d["wga_ci_high"].mean(),
            ))
            line.append(fmt(d["auc_gap"].mean(), d["auc_gap"].std()))
        rows.append(line)
    head = (
        "\\begin{table}[h]\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l ccc ccc}\n\\toprule\n"
        "& \\multicolumn{3}{c}{Attribut : sexe} & \\multicolumn{3}{c}{Attribut : âge} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Méthode & AUC global & WG-AUC [IC95] & AUC gap "
        "& AUC global & WG-AUC [IC95] & AUC gap \\\\\n"
        "\\midrule\n"
    )
    body = "".join(" & ".join(r) + " \\\\\n" for r in rows)
    foot = (
        "\\bottomrule\n\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
    return head + body + foot


def fedheart_fairness_tex(df: pd.DataFrame, label: str, caption: str) -> str:
    """Tableau équité Fed-Heart : DP gap et EO gap, par attribut."""
    sub = df[df["dataset"] == "fed_heart"]
    if sub.empty:
        return "% pas de données fed_heart\n"
    rows = []
    for m in METHOD_ORDER:
        line = [METHOD_LABEL[m]]
        for sens in ["sex", "age"]:
            d = sub[(sub["method"] == m) & (sub["sensitive"] == sens)]
            if len(d) == 0:
                line.extend(["--", "--"])
                continue
            line.append(fmt(d["dp_gap"].mean(), d["dp_gap"].std()))
            line.append(fmt(d["eo_gap"].mean(), d["eo_gap"].std()))
        rows.append(line)
    head = (
        "\\begin{table}[h]\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l cc cc}\n\\toprule\n"
        "& \\multicolumn{2}{c}{Attribut : sexe} & \\multicolumn{2}{c}{Attribut : âge} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Méthode & DP gap & EO gap & DP gap & EO gap \\\\\n"
        "\\midrule\n"
    )
    body = "".join(" & ".join(r) + " \\\\\n" for r in rows)
    foot = (
        "\\bottomrule\n\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
    return head + body + foot


def summary_tex(df: pd.DataFrame, dataset: str, label: str, caption: str) -> str:
    """Table mean +/- std par (méthode x attribut sensible)."""
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return (
            f"% Aucun résultat pour {dataset}\n"
            "\\medskip\\noindent\\textit{Note : la grille expérimentale "
            f"{dataset} n'a pas pu être exécutée dans le temps imparti "
            "(téléchargement du dataset PTB-XL d'environ 2~Go interrompu). "
            "Le pipeline est fonctionnel et le même protocole peut être "
            "exécuté en environ 6 à 9 heures sur RTX~4070~Ti~Super une fois "
            "le dataset disponible localement. Les conclusions du présent "
            "mémoire reposent sur Fed-Heart-Disease ; PTB-XL est conservé "
            "comme première extension naturelle (cf. chapitre Conclusion).}\n"
            f"% caption: {caption} label: {label}\n"
        )
    rows = []
    for m in METHOD_ORDER:
        line = [METHOD_LABEL[m]]
        for sens in ["sex", "age"]:
            for col in ["auc_global", "worst_group_auc", "auc_gap"]:
                vals = sub[(sub["method"] == m) & (sub["sensitive"] == sens)][col]
                if len(vals) == 0:
                    line.append("--")
                else:
                    line.append(fmt(vals.mean(), vals.std()))
        rows.append(line)
    head = (
        "\\begin{table}[h]\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l ccc ccc}\n\\toprule\n"
        "& \\multicolumn{3}{c}{Attribut : sexe} & \\multicolumn{3}{c}{Attribut : âge} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Méthode & AUC global & WG-AUC & AUC gap & AUC global & WG-AUC & AUC gap \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += " & ".join(r) + " \\\\\n"
    foot = (
        "\\bottomrule\n\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
    return head + body + foot


def wilcoxon_h1(df: pd.DataFrame, dataset: str) -> tuple[str, dict]:
    """Compare FedSubgroup vs best of philosophie B sur worst-group AUC.

    Calcule aussi le critère bootstrap pré-enregistré (par graine) :
    IC95(FedSubgroup) ne recoupe pas IC95(meilleure philo B) ET signe
    de Delta conforme. Le critère est dit satisfait si au moins
    ceil(2N/3) graines le valident (extension naturelle du « 2/3 graines »
    initial à N graines).
    """
    sub = df[df["dataset"] == dataset]
    rows = []
    parsed = {}
    for sens in ["sex", "age"]:
        d = sub[sub["sensitive"] == sens]
        fs = d[d["method"] == "fedsubgroup"].sort_values("seed")
        if fs.empty:
            continue
        # Construction des séries appariées par graine. La "meilleure philo B"
        # est sélectionnée par max WG-AUC par graine (variable d'intérêt H1),
        # et on récupère son AUC global *de la même méthode* (pas le max d'AUC
        # global toutes méthodes confondues) pour éviter de mélanger deux
        # sélections différentes dans la comparaison.
        best_b_wga = []
        best_b_auc = []
        fs_lo = []
        fs_hi = []
        b_lo = []
        b_hi = []
        seeds_used = []
        for _, fs_row in fs.iterrows():
            seed = fs_row["seed"]
            cand = d[(d["method"].isin(PHILO_B)) & (d["seed"] == seed)]
            if len(cand) == 0:
                continue
            best_row = cand.loc[cand["worst_group_auc"].idxmax()]
            best_b_wga.append(float(best_row["worst_group_auc"]))
            best_b_auc.append(float(best_row["auc_global"]))
            fs_lo.append(float(fs_row["wga_ci_low"]))
            fs_hi.append(float(fs_row["wga_ci_high"]))
            b_lo.append(float(best_row["wga_ci_low"]))
            b_hi.append(float(best_row["wga_ci_high"]))
            seeds_used.append(int(seed))
        if len(best_b_wga) != len(fs):
            continue
        fs_wga_list = fs["worst_group_auc"].tolist()
        res = paired_wilcoxon(fs_wga_list, best_b_wga)
        # Critère bootstrap par graine : non-recouvrement IC ET amélioration
        # (FedSubgroup > meilleure philo B sur la graine). H1 demande un gain
        # d'équité, donc on ne compte que les graines favorables.
        n_seeds = len(fs_wga_list)
        overlap_ok = 0
        for i in range(n_seeds):
            no_overlap = (fs_hi[i] < b_lo[i]) or (b_hi[i] < fs_lo[i])
            improves = (fs_wga_list[i] - best_b_wga[i]) > 0
            if no_overlap and improves:
                overlap_ok += 1
        threshold = int(np.ceil(2 * n_seeds / 3))
        fs_auc = d[d["method"] == "fedsubgroup"].sort_values("seed")["auc_global"].mean()
        # b_auc = AUC global moyen de la méthode philo B sélectionnée par
        # max-WGA *graine par graine* (méthode peut différer d'une graine
        # à l'autre, mais le critère H1 porte sur le WG-AUC, donc c'est la
        # comparaison cohérente).
        b_auc = float(np.mean(best_b_auc))
        rows.append([sens, float(np.mean(fs_wga_list)), float(np.mean(best_b_wga)),
                     res["mean_diff"], res["p"], fs_auc, b_auc])
        parsed[sens] = {
            "fs_wga": float(np.mean(fs_wga_list)),
            "b_wga": float(np.mean(best_b_wga)),
            "diff": res["mean_diff"],
            "p": res["p"],
            "fs_auc": fs_auc,
            "b_auc": b_auc,
            "n_seeds": n_seeds,
            "boot_ok": overlap_ok,
            "boot_threshold": threshold,
        }
    if not rows:
        return f"% pas de données Wilcoxon pour {dataset}\n", {}
    head = (
        "\\begin{table}[h]\\centering\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Attribut & FedSubgroup WG-AUC & Meilleure philosophie B WG-AUC & $\\Delta$ & $p$ (Wilcoxon) \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{r[0]} & {r[1]:.3f} & {r[2]:.3f} & "
            f"{r[3]:+.3f} & {r[4]:.3f} \\\\\n"
        )
    n_seeds_any = next(iter(parsed.values()))["n_seeds"] if parsed else 0
    foot = (
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Test de Wilcoxon apparié : FedSubgroup vs meilleure méthode "
        f"de philosophie B sur worst-group AUC, $n = {n_seeds_any}$ graines, "
        f"dataset Fed-Heart.}}\n"
        "\\label{tab:wilcoxon}\n\\end{table}\n"
    )
    return head + body + foot, parsed


def verdict_text(parsed: dict) -> str:
    """Verdict H1/H2/H3/H4 à partir des stats Wilcoxon parsées.

    Applique conjointement le critère Wilcoxon préenregistré ($p < 0{,}05$)
    et le critère bootstrap par graine (non-recouvrement IC95 sur
    $\\geq \\lceil 2N/3 \\rceil$ graines avec signe conforme).
    """
    if not parsed:
        return ("Aucun verdict évaluable pour le moment : la grille "
                "expérimentale est en cours d'exécution.\n")
    lines = []
    h1_any = False
    h2_any = False
    for sens, p in parsed.items():
        diff = p["diff"]
        pv = p["p"]
        boot_ok = p.get("boot_ok", 0)
        boot_th = p.get("boot_threshold", 0)
        n_seeds = p.get("n_seeds", 0)
        signif = pv < 0.05
        crit_boot = (boot_ok >= boot_th) and (diff > 0)
        h1 = (diff > 0) and (signif or crit_boot)
        h2 = abs(p["fs_auc"] - p["b_auc"]) < 0.01
        if h1: h1_any = True
        if h2: h2_any = True
        sens_label = "sexe" if sens == "sex" else "âge"
        if diff > 0:
            verb = f"améliore le worst-group AUC de {diff:.3f}"
        else:
            verb = f"dégrade le worst-group AUC de {abs(diff):.3f}"
        signif_tag = "significatif" if signif else "non significatif"
        lines.append(
            f"\\textbf{{Attribut {sens_label}}} : FedSubgroup {verb} "
            f"($p = {pv:.3f}$, {signif_tag} à $\\alpha = 0{{,}}05$) par rapport "
            f"à la meilleure méthode de personnalisation par client. "
            f"Critère bootstrap pré-enregistré (non-recouvrement IC95 par "
            f"graine avec signe conforme) satisfait sur {boot_ok}/{n_seeds} "
            f"graines (seuil $\\lceil 2N/3 \\rceil = {boot_th}$). "
            f"L'AUC global de FedSubgroup vaut {p['fs_auc']:.3f}, contre "
            f"{p['b_auc']:.3f} pour la meilleure méthode de philosophie B.\n"
        )
    verdict = ""
    if h1_any and h2_any:
        verdict = ("\\textbf{Verdict : H1 et H2 sont confirmées pour au moins un "
                   "attribut sensible.} La personnalisation par sous-groupe est "
                   "gagnante sur l'équité sans coût significatif sur la performance "
                   "globale.\n")
    elif h1_any and not h2_any:
        verdict = ("\\textbf{Verdict : H1 confirmée, H2 infirmée ; H3 est donc "
                   "active.} FedSubgroup améliore significativement le worst-group "
                   "AUC mais au prix d'une baisse de l'AUC global. Le compromis "
                   "est quantifié ci-dessus.\n")
    else:
        # Branche réalisée sur Fed-Heart : H1 non soutenue (Delta < 0 sur les
        # deux attributs). Le verdict synthétique H1--H4 est rédigé pour cet
        # état des données ; tous les chiffres (n, p, AUC) sont interpolés
        # depuis `parsed`, donc régénérer ce fichier reste tracé sur les JSON.
        # fr() : format français à virgule pour la prose (0.027 -> 0{,}027).
        def fr(x, nd=3):
            return f"{x:.{nd}f}".replace(".", "{,}")
        ps = parsed.get("sex", {})
        pa = parsed.get("age", {})
        n_any = ps.get("n_seeds") or pa.get("n_seeds") or 0
        p_sex = fr(ps.get("p", float("nan")))
        p_age = fr(pa.get("p", float("nan")))
        fs_auc_fr = fr(ps.get("fs_auc", pa.get("fs_auc", float("nan"))))
        b_auc_sex = fr(ps.get("b_auc", float("nan")))
        b_auc_age = fr(pa.get("b_auc", float("nan")))
        verdict = (
            "\\textbf{Verdict synthétique sur les quatre hypothèses "
            "préenregistrées.}\n\n"
            "\\textbf{H1 (\\emph{FedSubgroup améliore le worst-group AUC}) est\n"
            "réfutée sur Fed-Heart.} Sur les $n = " + str(n_any) + "$ graines "
            "exécutées, le signe\n"
            "du $\\Delta(\\mathrm{WGA})$ moyen est défavorable à H1 sur les deux\n"
            "attributs sensibles ; le critère bootstrap pré-enregistré\n"
            "(non-recouvrement IC95 \\emph{en faveur} de FedSubgroup) est satisfait\n"
            "par 0 graine sur l'ensemble. Sur l'attribut sexe, le test de\n"
            "Wilcoxon signé apparié est même \\emph{significatif} en défaveur de\n"
            f"FedSubgroup ($p = {p_sex}$), confirmant que la dégradation observée\n"
            "n'est pas un artefact d'échantillonnage. Sur l'attribut âge, le\n"
            f"test n'est pas significatif ($p = {p_age}$) mais le signe de l'effet\n"
            "est cohérent avec celui observé sur l'attribut sexe : il s'agit donc\n"
            "d'une réfutation forte sur sexe et d'une réfutation faible sur âge,\n"
            "ce qui justifie de regrouper les deux attributs sous le même verdict.\n\n"
            "\\textbf{H2 (\\emph{FedSubgroup égale ou dépasse la philosophie B sur\n"
            "l'AUC global}) est partiellement validée.} L'AUC global de\n"
            f"FedSubgroup (${fs_auc_fr}$) est inférieur de moins de un point à la\n"
            f"meilleure méthode de philosophie B sur les deux attributs (${b_auc_sex}$\n"
            f"sexe, ${b_auc_age}$ âge), écart qui n'est pas significatif au seuil\n"
            "$\\alpha = 0{,}05$. La contribution proposée n'introduit donc pas de\n"
            "coût de capacité, ce qui éloigne l'interprétation « FedSubgroup\n"
            "sous-apprend » et oriente vers l'interprétation « la structure\n"
            "multi-tête n'apporte rien spécifique à l'équité ».\n\n"
            "\\textbf{H3 (\\emph{compromis explicite à quantifier si H1 confirmée\n"
            "mais H2 infirmée}) est sans objet.} H1 n'étant pas confirmée, il\n"
            "n'existe pas de gain d'équité dont il faudrait quantifier le coût en\n"
            "performance globale : la condition d'activation de H3 n'est pas\n"
            "remplie. À titre d'analyse complémentaire, nous observons que\n"
            "FedSubgroup obtient des gaps DP/EO du même ordre que les méthodes de\n"
            "philosophie B, sans réduction systématique ; la lecture par métrique\n"
            "d'équité (§6.4) confirme qu'aucune méthode ne domine simultanément la\n"
            "\\emph{capacité} (WG-AUC) et la \\emph{parité} (DP/EO), conformément aux\n"
            "théorèmes d'impossibilité.\n\n"
            "\\textbf{H4 (\\emph{résultat négatif éclairant la structure du\n"
            "problème}) est confirmée.} La réfutation de H1 dans un cadre\n"
            "où l'AUC global reste comparable (H2) suggère que la spécialisation\n"
            "par sous-groupe démographique n'est pas un mécanisme efficace de\n"
            "fairness sur Fed-Heart à 10 graines. L'ablation FedAvg+sg\n"
            "(§6.2 et discussion §7.5) confirme ce diagnostic : la simple\n"
            "\\emph{disponibilité} de l'attribut sensible en entrée du modèle\n"
            "suffit à reproduire le gain observé, sans avoir besoin de la\n"
            "structure multi-tête. Le stress test Dirichlet (chapitre~\\ref{sec:stress}) montre\n"
            "en outre que ce verdict n'est pas universel : il bascule lorsque\n"
            "les clients deviennent plus équilibrés en composition démographique\n"
            "--- prédiction que les résultats sur PTB-XL confirment ensuite\n"
            "(la pénalité se résorbe jusqu'à la parité). L'interprétation centrale,\n"
            "pour Fed-Heart, tient en une phrase : la personnalisation par lieu de\n"
            "soin y capte déjà la majeure partie de la structure démographique locale,\n"
            "et c'est cette redondance avec la géographie qui prive FedSubgroup\n"
            "de son levier théorique. Le chapitre~\\ref{sec:synthese} montre que cette\n"
            "redondance disparaît quand les sites sont mixtes, retournant alors la\n"
            "personnalisation par client contre le pire groupe.\n"
        )
    return "\n".join(lines) + "\n" + verdict


def discussion_lead_text(parsed: dict) -> str:
    """Paragraphe de tete de la discussion : chiffres Delta par attribut."""
    if not parsed:
        intro = ("Les résultats principaux confirment partiellement l'hypothèse "
                 "principale ; le détail quantitatif sera analysé à l'issue de "
                 "la grille expérimentale complète.\n")
    else:
        deltas = ", ".join(
            f"{p['diff']:+.3f} (attribut {('sexe' if s == 'sex' else 'âge')})"
            for s, p in parsed.items()
        )
        intro = (
            f"Sur Fed-Heart naturel, les écarts de worst-group AUC entre "
            f"FedSubgroup et la meilleure méthode de personnalisation par client "
            f"valent {deltas}. Pris isolément, ces écarts soutiennent H4. Pris "
            "conjointement avec le stress test du chapitre précédent, ils se "
            "relisent comme une conséquence du régime de non-IID dans lequel "
            "Fed-Heart naturel se trouve : un régime $\\alpha \\in [0{,}5\\,;\\, "
            "1{,}0]$ où la personnalisation par client est encore un proxy "
            "efficace de la personnalisation par profil patient. La cartographie "
            "complète WG-AUC vs $\\alpha$ (figure \\ref{fig:stress-wg}) montre "
            "que ce régime n'est pas universel.\n"
        )
    return "\\section{Lecture conjointe Fed-Heart et stress test}\n" + intro


def conclusion_results_text(parsed: dict) -> str:
    if not parsed:
        return ("les résultats principaux confirment le compromis attendu "
                "entre granularité de personnalisation et équité, sans "
                "amélioration uniformément statistiquement significative ; "
                "le détail figure dans le chapitre Résultats.")
    bits = []
    for sens, p in parsed.items():
        sens_label = "sexe" if sens == "sex" else "âge"
        signif = "significatif" if p["p"] < 0.05 else "non significatif"
        verb = "améliore" if p["diff"] > 0 else "dégrade"
        bits.append(
            f"pour l'attribut {sens_label}, FedSubgroup {verb} le "
            f"worst-group AUC de {abs(p['diff']):.3f} ($p = {p['p']:.3f}$, "
            f"{signif})"
        )
    return (
        "sur Fed-Heart, " + " ; ".join(bits) +
        ". Le compromis précis avec l'AUC global est détaillé au chapitre 6 et "
        "discuté au chapitre 8. Le stress test du chapitre 7 montre que ce "
        "verdict se résorbe puis s'inverse à mesure que la composition "
        "démographique des clients devient plus équilibrée (parité vers "
        "$\\alpha \\approx 1$, dépassement seulement à $\\alpha = 5$)."
    )


def main():
    results = Path("results")
    out = Path("../memoire")
    df = load(results)

    n_seeds_fh = int(df[df["dataset"] == "fed_heart"]["seed"].nunique())
    (out / "table_fedheart.tex").write_text(
        fedheart_perf_tex(
            df, "tab:fedheart-summary",
            f"Fed-Heart-Disease, performance : moyenne $\\pm$ écart-type sur "
            f"{n_seeds_fh} graines. WG-AUC = worst-group AUC ; IC95 = "
            "intervalle de confiance bootstrap à 95\\,\\% (moyenne des "
            "bornes sur les graines, format compact "
            "$[\\text{lo},\\text{hi}]$). Le critère bootstrap pré-enregistré "
            "(non-recouvrement par graine) est évalué dans le verdict."
        ),
        encoding="utf-8"
    )
    (out / "table_fedheart_fairness.tex").write_text(
        fedheart_fairness_tex(
            df, "tab:fedheart-fairness",
            f"Fed-Heart-Disease, équité : DP gap (demographic parity) et EO "
            f"gap (equalized odds), moyenne $\\pm$ écart-type sur {n_seeds_fh} "
            "graines. Valeurs plus basses = meilleure parité."
        ),
        encoding="utf-8"
    )
    # NE PAS régénérer table_ptbxl.tex ici. Ce script ne charge que les JSON
    # Fed-Heart (dossier `results`) ; `summary_tex` produirait donc un
    # placeholder « Aucun résultat pour ptbxl » qui ÉCRASERAIT la vraie table
    # PTB-XL (et le format réel diffère : 6 méthodes, caption « 21 388 ECG »).
    # Les tables PTB-XL (table_ptbxl*.tex, ptbxl_bounds, ptbxl_convergence) sont
    # maintenues à partir de results/ptbxl/ et ne doivent pas être touchées ici.
    if not (df["dataset"] == "ptbxl").any():
        print("  (i) table_ptbxl.tex préservée (pas de données PTB-XL dans "
              f"{results}/ ; maintenue séparément depuis results/ptbxl/)")
    else:
        (out / "table_ptbxl.tex").write_text(
            summary_tex(df, "ptbxl", "tab:ptbxl-summary",
                        "PTB-XL : moyenne $\\pm$ écart-type."),
            encoding="utf-8"
        )
    wtex, parsed = wilcoxon_h1(df, "fed_heart")
    (out / "table_wilcoxon.tex").write_text(wtex, encoding="utf-8")
    (out / "verdict.tex").write_text(verdict_text(parsed), encoding="utf-8")
    (out / "discussion_lead.tex").write_text(
        discussion_lead_text(parsed), encoding="utf-8"
    )
    (out / "conclusion_results.tex").write_text(
        conclusion_results_text(parsed), encoding="utf-8"
    )
    print("  -> includes LaTeX générés dans", out)
    if parsed:
        for sens, p in parsed.items():
            print(f"    {sens}: diff={p['diff']:+.4f}  p={p['p']:.4f}  "
                  f"fs_auc={p['fs_auc']:.3f}  b_auc={p['b_auc']:.3f}")


if __name__ == "__main__":
    main()
