"""Every number and table of the paper's results, generated from the result files (no hand-copied numbers).

    python tools/paper_numbers.py        # -> paper_gen/numbers.tex (macros) and paper_gen/tab_*.tex

paper_needed.tex \\input's these files. Order of the pipeline (README.md, README.md):
  proxy_verify.py variants / extract -> proxy_issuer_eval.py evaluate -> proxy_paper_numbers.py -> proxy_referee_numbers.py -> this script.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_gen"


def J(name):
    return json.loads((ROOT / "outputs" / name).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def kn(r):
    return f"{r['k']}/{r['n']}"


def main():
    OUT.mkdir(exist_ok=True)
    R = J("proxy_attr_issuer_results.json")
    r = R["results"]
    P = J("proxy_attr_paper_numbers.json")
    N = J("proxy_attr_referee_numbers.json")
    B = J("baseline_behavioural.json")["rows"]
    Qa, Qn = J("quality_paired_proxy_ravdess.json"), J("quality_paired_proxy_nir.json")
    plan = J("proxy_attr_plan4.json") if (ROOT / "outputs/proxy_attr_plan4.json").exists() else J("proxy_attr_plan3.json")
    enrol = plan["enrolment"]
    rows = R["rows"]
    claims = R["claims"]
    M = {}

    def m(name, value):
        M[name] = str(value)

    # ---------------------------------------------------------------- people and sessions
    m("nPeople", len(enrol))
    m("nRavPeople", sum(e["dataset"] == "ravdess" for e in enrol.values()))
    m("nNirPeople", sum(e["dataset"] == "nir" for e in enrol.values()))
    m("nOthers", len(enrol) - 1)
    m("nPool", plan["pool_size"])
    gen_h = [x for x in rows if x["variant"] == "original" and x["kind"] in ("self", "cross") and x.get("generator", "liveportrait") == "liveportrait"]
    m("nGenerated", len(gen_h))
    m("nTuning", r["tuning_honest"]["n"])
    # the threshold was frozen when only the first people were enrolled: its impostors are the other proxies of that first plan, not of the final ledger
    try:
        _sc = J("scores_liveportrait.json")
        _frozen = {v["did"] for v in J("proxy_attr_plan.json")["enrolment"].values()} & set(_sc["proxies"])
    except FileNotFoundError:
        _frozen = None
    nfz = len(_frozen) if _frozen else len(enrol)
    m("nFreezeProxies", nfz)
    m("nTuningImp", r["tuning_honest"]["n"] * (nfz - 1))
    h = r["honest_all"]
    m("nTest", h["n"])
    m("nTestSelf", r["honest_self"]["n"])
    m("nTestCross", r["honest_cross"]["n"])
    m("nTestRav", r["honest_ravdess"]["n"])
    m("nTestNir", r["honest_nir"]["n"])
    plan1_ids = {x["id"] for x in J("proxy_attr_plan.json")["rows"]}
    test_h = [x for x in gen_h if not x["tuning"]]
    m("nAfterFreeze", sum(x["id"] not in plan1_ids for x in test_h))                      # generated after the threshold was frozen
    m("nFirstBatchTest", sum(x["id"] in plan1_ids for x in test_h))
    m("nBatchFour", sum(("cross4_" in x["id"] or "self4_" in x["id"]) for x in test_h))
    # ---------------------------------------------------------------- verdicts
    c = h["correct"]
    m("kCorrect", c["k"])
    m("pCorrect", pct(c["rate"]))
    m("pCorrectInt", f"{100 * c['rate']:.0f}")
    m("ciCorrectLo", pct(c["ci95"][0], 0))
    m("ciCorrectHi", pct(c["ci95"][1], 0))
    m("nWrong", h["wrong"]["k"])
    m("wrongUpper", pct(h["wrong"]["ci95"][1]))
    m("nRefused", h["inconclusive"]["k"])
    ho = r["honest_outcomes"]
    m("nNotCorrect", h["n"] - c["k"])
    m("pNotCorrectInt", f"{100 * (h['n'] - c['k']) / h['n']:.0f}")
    m("nClaimFailed", ho.get("CLAIM_FAILED", 0))
    m("nFlagged", ho.get("FLAGGED", 0))
    m("pRefusedInt", f"{100 * h['inconclusive']['rate']:.0f}")
    for key, name in (("honest_self", "Self"), ("honest_cross", "Cross"), ("honest_ravdess", "Rav"), ("honest_nir", "Nir")):
        m(f"rate{name}", kn(r[key]["correct"]))
        m(f"pct{name}", pct(r[key]["correct"]["rate"]))
    # ---------------------------------------------------------------- claims
    fc = r["false_claims"]
    m("nFalse", sum(v["n"] for v in fc.values()))
    m("nFalseAccepted", sum(v["accepted_as_self_or_cross"]["k"] for v in fc.values()))
    n_f = sum(v["n"] for v in fc.values())
    m("falseUpper", pct(3.0 / n_f if n_f else 0, 1))     # rule of three, 95 % upper bound for 0 events
    m("nFalseLiveIris", fc.get("driver's key, another person's iris", {}).get("accepted_as_self_or_cross", {}).get("k", 0))
    m("nFalseLiveIrisN", fc.get("driver's key, another person's iris", {}).get("n", 0))
    m("nFalseOther", sum(v["accepted_as_self_or_cross"]["k"] for k, v in fc.items() if k != "driver's key, another person's iris"))
    lv = r["issuer_live_iris_check"]
    m("liveGen", kn(lv["genuine"]))
    m("liveImp", kn(lv["impostor"]))
    # ---------------------------------------------------------------- distance
    sep = P["separation_test"]["arciris"]
    bs = N["bootstrap_arciris_1000"]
    op = N["operating"]
    m("eer", sep["eer"]); m("eerLo", bs["eer"][0]); m("eerHi", bs["eer"][1])
    m("auc", sep["auc"]); m("aucLo", bs["auc"][0]); m("aucHi", bs["auc"][1])
    m("tar", sep["tar_far1"]); m("tarLo", bs["tar_far1"][0]); m("tarHi", bs["tar_far1"][1])
    m("lgEer", P["separation_test"]["irislg-cons"]["eer"])
    m("nImpostors", sep["n_impostor"])
    m("Tone", f"{op['T1_frozen']:.4f}")
    m("distAcc", kn(op["distance_only_accepted_at_T1"]))
    m("impAcc", kn(op["impostor_accepted_at_T1"]))
    m("impAccPct", pct(op["impostor_accepted_at_T1"]["rate"]))
    # ---------------------------------------------------------------- record-only arm
    ro = N["record_only"]
    m("nTwin", ro["twin_without_hook"]["n"])
    m("twinRecordOnly", ro["twin_without_hook"]["record_only_attributed"]["k"])
    m("nHonestRecordOnly", ro["honest"]["record_only_attributed"]["k"])
    m("nHonestToRefusal", ro["honest"]["record_only_attributed"]["k"] - ro["honest"]["full_attributed"]["k"])
    bj = ROOT / "outputs" / "behavioural_on_sessions.json"
    if bj.exists():
        pi = json.loads(bj.read_text())["models"]["proxy_iris"]["per_generator"]
        eo = [pi[g]["proxy_distance"]["eer"] for g in ("personalive", "liax", "edtalk") if g in pi]
        m("eerOtherLo", f"{min(eo):.1f}"); m("eerOtherHi", f"{max(eo):.1f}")
        bm_ = json.loads(bj.read_text())["models"]
        for k_, nm_ in (("trustfa", "Trust"), ("graph", "Graph"), ("dino", "Dino"), ("clip", "Clip"), ("fusion", "Fusion")):
            m("behAuc" + nm_, "%.1f" % bm_[k_]["per_generator"]["liveportrait"]["reference"]["auc"])
        for g_, nm_ in (("personalive", "Pl"), ("liax", "Liax"), ("edtalk", "Ed")):
            if g_ in pi:
                m("eer" + nm_, "%.1f" % pi[g_]["proxy_distance"]["eer"]); m("auc" + nm_, "%.1f" % pi[g_]["proxy_distance"]["auc"])
    bi = ROOT / "outputs" / "behavioural_imitation" / "results.json"
    if bi.exists():
        bm = json.loads(bi.read_text())["models"]
        au = [v["b_claim_as_D"]["imitation_vs_original_by_claim_score"]["auc"] for v in bm.values()]
        m("imitBehAucLo", f"{min(au):.1f}"); m("imitBehAucHi", f"{max(au):.1f}")
    m("nStolenKey", N["claim_arms"]["stolen_key_n"])
    if "rank1" in N:
        m("rankOne", pct(N["rank1"]["rate"])); m("rankOneK", N["rank1"]["k"]); m("rankOneN", N["rank1"]["n"])
    # ---------------------------------------------------------------- imitation
    im = P["imitation"]
    m("nImit", im["n"])
    m("imitAttr", kn(im["attacker_attributed"]))
    m("imitAcc", kn(im["imitated_person_accepted"]))
    m("imitAccK", im["imitated_person_accepted"]["k"])
    m("imitUpper", pct(im["imitated_person_accepted"]["ci95"][1], 0) if im["imitated_person_accepted"]["ci95"] else "n/a")
    ra = R.get("record_attacks", {})
    m("openProxyRej", ra.get("issuer_opens_another_proxy", {}).get("outcomes", {}).get("REJECT", 0))
    m("openSaltRej", ra.get("issuer_opens_with_a_wrong_salt", {}).get("outcomes", {}).get("REJECT", 0))
    # ---------------------------------------------------------------- robustness
    for v in ("crf23", "crf30", "blur", "alter"):
        x = r[f"variant_{v}"]
        att = x["outcomes"].get("SELF", 0) + x["outcomes"].get("CROSS", 0)
        nm = dict(crf23="CrfLow", crf30="CrfHigh", blur="Blur", alter="Alter")[v]       # control-sequence names cannot hold digits
        m(f"var{nm}N", x["n"]); m(f"var{nm}Att", att)
    m("crfAuc", N["after_crf23"]["roc"]["auc"]); m("crfTar", f"{N['after_crf23']['roc']['tar_far1']:.0f}")
    # ---------------------------------------------------------------- pool separation, overhead, quality
    pw, same = N["pool_pairwise_arciris"], N["pool_clean_same_eye_other_capture"]
    m("poolPairs", pw["n_pairs"]); m("poolFirstPct", f"{pw['p1']:.2f}"); m("sameN", same["n"]); m("sameMax", f"{same['max']:.2f}"); m("nUnusedPool", plan["pool_size"] - len(enrol))
    m("poolMed", f"{pw['median']:.2f}"); m("poolMin", f"{pw['min']:.2f}"); m("sameMed", f"{same['median']:.2f}"); m("sameNinety", f"{same['p90']:.2f}")
    m("genMed", f"{op['genuine_distance']['median']:.2f}"); m("impPFirst", f"{op['impostor_distance']['p1']:.2f}"); m("impMin", f"{op['impostor_distance']['min']:.2f}")
    ov = P["overhead"]
    m("hookRav", f"{ov['embed_ms_per_frame_median']['ravdess']:.0f}"); m("hookNir", f"{ov['embed_ms_per_frame_median']['nir']:.0f}")
    m("sessRav", f"{ov['session_seconds_per_frame_median']['ravdess']:.2f}"); m("sessNir", f"{ov['session_seconds_per_frame_median']['nir']:.2f}")
    m("extractSec", f"{ov['extraction_seconds_per_video_median']:.0f}"); m("verifySec", f"{ov['verify_seconds_after_extraction_median']:.1f}")
    m("pdqSec", f"{ov['pdq_seconds_per_video']:.1f}"); m("recordBytes", ov["record_bytes"])
    m("qLpipsRav", f"{Qa['summary']['lpips']['mean']:.4f}"); m("qPsnrRav", f"{Qa['summary']['psnr']['mean']:.1f}")
    m("qEyeLpipsRav", f"{Qa['summary']['eye_lpips']['mean']:.3f}"); m("qEyeLpipsNir", f"{Qn['summary']['eye_lpips']['mean']:.3f}")
    bg_ = r.get("by_generator", {})
    for g, nm in (("personalive", "Pl"), ("liax", "Liax"), ("edtalk", "Ed"), ("liveportrait", "Lp")):
        x = bg_.get(g)
        if x:
            m(f"n{nm}", x["n"]); m(f"k{nm}Correct", x["correct"]["k"]); m(f"p{nm}Correct", pct(x["correct"]["rate"], 0)); m(f"n{nm}Wrong", x["wrong"]["k"])
            m(f"n{nm}Refused", x["refused"]["k"]); m(f"n{nm}Twin", x["twin_attributed"]["n"]); m(f"k{nm}Twin", x["twin_attributed"]["k"])
    # ---------------------------------------------------------------- baselines
    m("bTrustEer", f"{B['trustfa_nvfair']['eer']:.1f}"); m("bGraphEer", f"{B['graph_ravdess_liveportrait']['eer']:.1f}")
    (OUT / "numbers.tex").write_text("% generated by tools/paper_numbers.py; do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")

    # ---------------------------------------------------------------- tables
    def tab(name, body):
        (OUT / f"tab_{name}.tex").write_text(body + "\\bottomrule\n")      # \bottomrule inside the file: after \input an EOF space token would make it misplaced
    # attribution table
    tab("attr", "\n".join([
        r"Inter-frame differencing~\cite{chapariniya2026micro} (retrained) & post hoc & %.1f & %.1f & %.1f & -- \\" % (B["trustfa_nvfair"]["auc"], B["trustfa_nvfair"]["eer"], B["trustfa_nvfair"]["tar_far1pct"]),
        r"Motion GCN~\cite{pedrouzo2025really} & post hoc & %.1f & %.1f & %.1f & -- \\" % (B["graph_ravdess_liveportrait"]["auc"], B["graph_ravdess_liveportrait"]["eer"], B["graph_ravdess_liveportrait"]["tar_far1pct"]),
        r"Ours, ArcIris (primary) & generation & %.1f & %.1f & %.1f & %s \\" % (sep["auc"], sep["eer"], sep["tar_far1"], pct(c["rate"])),
        r"\textbf{Ours, log-Gabor} (second extractor) & generation & %.1f & %.1f & %.1f & -- \\" % (P["separation_test"]["irislg-cons"]["auc"], P["separation_test"]["irislg-cons"]["eer"], P["separation_test"]["irislg-cons"]["tar_far1"])]) + "\n")

    CA = N["claim_arms"]
    def trow(name, key):
        x = ro[key]
        return r"%s & %d & %d & %d & %d \\" % (name, x["n"], x["key_record_attributed"]["k"], x["record_only_attributed"]["k"], x["full_attributed"]["k"])
    tab("added", "\n".join([trow("Honest session", "honest"), trow("Same generation without the hook", "twin_without_hook"), trow("Faulty issuer embeds the source's proxy", "wrong_proxy_embedded"),
                            trow("Faulty issuer embeds an unenrolled proxy", "unenrolled_proxy_embedded"), trow("Re-encoded, CRF 23", "crf23"), trow("Re-encoded, CRF 30", "crf30"),
                            trow("Blurred frames", "blur"), trow("Edited (lower face covered)", "alter"), r"\midrule",
                            r"Claim with the driver's key and another person's iris & %d & %d & %d & %d \\" % (CA["stolen_key_n"], CA["stolen_key_accepted"]["key_record"], CA["stolen_key_accepted"]["live_iris"], CA["stolen_key_accepted"]["full"]),
                            r"Other false claims (wrong key, other person, photo owner) & %d & %d & %d & %d \\" % (CA["other_false_claims_n"], 0, 0, CA["other_false_accepted"]["full"]),
                            r"Imitation: the imitated person claims the attacker's video & %d & 0 & 0 & %s \\" % (im["n"], im["imitated_person_accepted"]["k"])]) + "\n")
    tab("quality", "\n".join([
        r"RAVDESS & %d & %.4f & %.3f & %.1f & %.4f & %.2f & %.1f \\" % (Qa["summary"]["lpips"]["n_clips"], Qa["summary"]["lpips"]["mean"], Qa["summary"]["eye_lpips"]["mean"], Qa["summary"]["psnr"]["mean"], Qa["summary"]["ssim"]["mean"], Qa["summary"]["akd_pct_iod"]["mean"], Qa["eye_crop_fid"]),
        r"NIR & %d & %.4f & %.3f & %.1f & %.4f & %.2f & %.1f \\" % (Qn["summary"]["lpips"]["n_clips"], Qn["summary"]["lpips"]["mean"], Qn["summary"]["eye_lpips"]["mean"], Qn["summary"]["psnr"]["mean"], Qn["summary"]["ssim"]["mean"], Qn["summary"]["akd_pct_iod"]["mean"], Qn["eye_crop_fid"])]) + "\n")
    labels = dict(crf23="Re-encoded, CRF 23", crf30="Re-encoded, CRF 30", blur="Blurred frames", alter="Edited video (lower face covered)")
    lines = []
    for v in ("crf23", "crf30", "blur", "alter"):
        x = r[f"variant_{v}"]
        att = x["outcomes"].get("SELF", 0) + x["outcomes"].get("CROSS", 0)
        lines.append(r"%s & %d & %d (%.0f\%%) & %d & %d & %d \\" % (labels[v], x["n"], att, 100 * att / x["n"], x["outcomes"].get("INCONCLUSIVE", 0), x["outcomes"].get("REJECT", 0), x["outcomes"].get("CLAIM_FAILED", 0)))
    tab("robust", "\n".join(lines) + "\n")
    ab = P["ablation"]
    fr = ab["frames_fused_arciris"]
    a = lambda name, x: r"%s & %.1f & %.1f \\" % (name, 100 * x["genuine"]["rate"], 100 * x["impostor"]["rate"])
    tab("ablation", "\n".join([r"Deployed verifier (frozen $T_1$, 5 or more frames) & %.1f & %.1f \\" % (100 * op["distance_only_accepted_at_T1"]["rate"], 100 * op["impostor_accepted_at_T1"]["rate"]),
                               a("Re-read threshold, all analysed frames", fr["all"]), a("Frames fused: 1 analysed frame", fr["1"]), a("Frames fused: 16", fr["16"]),
                               a("Extractor: log-Gabor code, majority vote", ab["extractor"]["irislg-cons"])]) + "\n")
    m("ablDep", "%.1f / %.1f" % (100 * op["distance_only_accepted_at_T1"]["rate"], 100 * op["impostor_accepted_at_T1"]["rate"]))
    for k_, nm_ in (("all", "All"), ("1", "One"), ("16", "Sixteen")):
        m("abl" + nm_, "%.1f / %.1f" % (100 * fr[k_]["genuine"]["rate"], 100 * fr[k_]["impostor"]["rate"]))
    m("ablLg", "%.1f / %.1f" % (100 * ab["extractor"]["irislg-cons"]["genuine"]["rate"], 100 * ab["extractor"]["irislg-cons"]["impostor"]["rate"]))
    bg = r.get("by_generator", {})
    names = dict(liveportrait="LivePortrait", personalive="PersonaLive", liax="LIA-X", edtalk="EDTalk")
    gl = []
    for g in ("liveportrait", "personalive", "liax", "edtalk"):
        if g in bg:
            x = bg[g]
            gl.append(r"%s & %d & %d (%.0f\%%) & %d & %d & %d \\" % (names[g], x["n"], x["correct"]["k"], 100 * x["correct"]["rate"], x["wrong"]["k"], x["refused"]["k"], x["twin_attributed"]["k"]))
    gl = gl or ["(no sessions yet) & & & & & \\\\"]
    tab("generators", "\n".join(gl) + "\n")
    # ---------------------------------------------------------------- resolution sweep and appearance change (README.md)
    RS = json.loads((ROOT / "outputs" / "resolution_sweep" / "results.json").read_text()) if (ROOT / "outputs" / "resolution_sweep" / "results.json").exists() else None
    AP = json.loads((ROOT / "outputs" / "appearance_change" / "results.json").read_text()) if (ROOT / "outputs" / "appearance_change" / "results.json").exists() else None
    if RS:
        def rs(ds, arm, target):
            for x in RS["by_dataset"][ds]:
                if x["arm"] == arm and (target is None or int(x["target"]) == target):
                    return x
        def cell(x):
            return "%.0f / %.1f" % (100 * x["genuine_accepted_at_T1"]["rate"], x["roc"]["eer"])
        rows = [r"Original files (%.0f / %.0f px) & %s & %s & %s & %s \\" % (RS["native_diameter_px"]["ravdess"]["median"], RS["native_diameter_px"]["nir"]["median"],
                cell(rs("ravdess", "native", None)), "--", cell(rs("nir", "native", None)), "--")]
        for px in (60, 40, 30, 20):
            rows.append(r"%d px & %s & %s & %s & %s \\" % (px, cell(rs("ravdess", "crf18", px)), cell(rs("ravdess", "lossless", px)), cell(rs("nir", "crf18", px)), cell(rs("nir", "lossless", px))))
        tab("resolution", "\n".join(rows) + "\n")
        m("resNRav", RS["by_dataset"]["ravdess"][0]["n_sessions"]); m("resNNir", RS["by_dataset"]["nir"][0]["n_sessions"])
        m("resRawRav", RS["raw_twin_control"]["ravdess"]["genuine_accepted_at_T1"]["k"]); m("resRawNir", RS["raw_twin_control"]["nir"]["genuine_accepted_at_T1"]["k"])
        m("resNativeRav", "%.0f" % RS["native_diameter_px"]["ravdess"]["median"]); m("resNativeNir", "%.0f" % RS["native_diameter_px"]["nir"]["median"])
    if AP:
        ar, an = AP["by_dataset"]["ravdess"], AP["by_dataset"]["nir"]
        m("dEMean", "%.1f" % ar["session_dE_mean"]["mean"]); m("dEMedian", "%.1f" % ar["session_dE_mean"]["median"]); m("dEPNinety", "%.1f" % ar["session_dE_mean"]["p90"])
        m("dEShareFive", "%.0f" % (100 * ar["share_sessions_dE_mean_gt5"]["rate"])); m("dEShareTen", "%.0f" % (100 * ar["share_sessions_dE_mean_gt10"]["rate"]))
        m("dEN", ar["n_sessions"]); m("dELight", "%.1f" % ar["session_dL_signed"]["mean"]); m("dENir", "%.1f" % an["session_dE_mean"]["mean"])
        for g, nm in (("personalive", "Pl"), ("liax", "Liax"), ("edtalk", "Ed")):
            m("dE" + nm, "%.1f" % AP["by_generator"][g]["session_dE_mean"]["mean"])
        RF = AP["reference_scales"]
        m("dETwin", "%.1f" % RF["twin_vs_twin_next"]["ravdess"]["mean"]); m("dESource", "%.1f" % RF["source_vs_twin"]["liveportrait/ravdess"]["mean"])
        m("dEPeople", "%.1f" % RF["between_people"]["ravdess"]["mean"]); m("dEProxies", "%.1f" % RF["two_proxies_same_frames"]["pair_dE_mean"]["mean"])
        m("dEProxiesShare", "%.0f" % (100 * RF["two_proxies_same_frames"]["share_pairs_dE_mean_gt5"]))
    CM_F = ROOT / "outputs" / "colour_match" / "results.json"
    if CM_F.exists():
        CM = json.loads(CM_F.read_text())
        cs, at = CM["colour_shift"], CM["attribution"]
        m("cmN", at["n_sessions"])
        m("cmDeDonor", "%.1f" % cs["donor"]["dE_mean_session"]["mean"]); m("cmDeTarget", "%.1f" % cs["target"]["dE_mean_session"]["mean"])
        m("cmTargetGtFive", cs["target"]["share_dE_mean_gt5"]["k"]); m("cmDonorGtFive", cs["donor"]["share_dE_mean_gt5"]["k"])
        m("cmPairedDe", "%.1f" % abs(cs["paired_dE_mean_target_minus_donor"]["mean"]))
        m("cmAccDonor", at["donor"]["genuine_accepted_at_T1_of_all"]["k"]); m("cmAccTarget", at["target"]["genuine_accepted_at_T1_of_all"]["k"])
        m("cmEerDonor", "%.1f" % at["donor"]["EER_percent_vs_other_proxies"]); m("cmEerTarget", "%.1f" % at["target"]["EER_percent_vs_other_proxies"])
        m("cmOnlyDonor", at["paired_acceptance"]["only_donor"]); m("cmOnlyTarget", at["paired_acceptance"]["only_target"])
    WM = json.loads((ROOT / "outputs" / "watermark_study" / "results.json").read_text()) if (ROOT / "outputs" / "watermark_study" / "results.json").exists() else None
    if WM:
        names_ = dict(clean="None (released video)", crf23="H.264 CRF 23", crf30="H.264 CRF 30", blur=r"Blur, $\sigma=14$", resize05="Resize to 0.5$\\times$", crop08="Zoom crop 0.8")
        rows = []
        for cnd in ("clean", "crf23", "crf30", "blur", "resize05", "crop08"):
            x = WM["table"][cnd]; n_ = WM["watermark_arm"]["n_sessions"]
            rows.append(r"%s & %d / %d & %.0f / %.1f \\" % (names_[cnd], round(x["wm_session_recovery"] * n_), n_, 100 * x["iris_genuine_acceptance"], x["iris_eer_percent"]))
        tab("watermark", "\n".join(rows) + "\n")
        m("wmN", WM["watermark_arm"]["n_sessions"]); m("wmCombN", WM["combined_arm"]["n_sessions"])
        for cnd_, nm_ in (("clean", "Clean"), ("crf23", "CrfLow"), ("crf30", "CrfHigh"), ("blur", "Blur"), ("resize05", "Resize"), ("crop08", "Crop")):
            m("wmAcc" + nm_, "%.0f" % (100 * WM["table"][cnd_]["iris_genuine_acceptance"])); m("wmEer" + nm_, "%.1f" % WM["table"][cnd_]["iris_eer_percent"])
        q = WM["quality"]
        m("wmPsnr", "%.1f" % q["watermark_vs_raw"]["psnr_db"]["mean"]); m("wmLpips", "%.4f" % q["watermark_vs_raw"]["lpips"]["mean"]); m("wmChanged", "%.0f" % (100 * q["watermark_vs_raw"]["changed_pixel_fraction"]["mean"]))
        m("irisPsnrW", "%.1f" % q["iris_vs_raw"]["psnr_db"]["mean"]); m("irisPsnrBox", "%.1f" % q["iris_vs_raw"]["psnr_changed_bbox_db"]["mean"]); m("irisChanged", "%.0f" % (100 * q["iris_vs_raw"]["changed_pixel_fraction"]["mean"]))
        m("wmFalse", "%.0f" % (100 * WM["table"]["clean"]["wm_false_lookup_unwatermarked"]))
    (OUT / "numbers.tex").write_text("% generated by tools/paper_numbers.py; do not edit\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    print("wrote", len(M), "macros and", len(list(OUT.glob("tab_*.tex"))), "tables to", OUT)


if __name__ == "__main__":
    main()
