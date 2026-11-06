from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toscana.data.xy import DataXYE
from toscana.numerics.fitting import fittingRange
from toscana.models.ni import niPeaks10
from toscana.physics.crystallography import NI_LATTICE_PARAMETER, reflections_fcc
from toscana.utils.dates import getDate

if TYPE_CHECKING:
    import lmfit as lm  # type: ignore[import-untyped]


def setting_model_d4nifit(  # noqa: N802 (legacy API)
    nifile: str, zac_ini: float = 0.0, lattice: float = NI_LATTICE_PARAMETER
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[str],
    list[float],
    DataXYE,
]:
    """
    Prepare initial parameters and plotting limits for Ni powder calibration.

    `lattice` is the Ni lattice parameter in Å (default 3.52 Å).
    """
    import matplotlib.pyplot as plt

    nickel = DataXYE(nifile)
    nickel.plot()

    wl_ini = 2 * lattice / np.sqrt(3.0) * np.sin(nickel.peaks_x[0] * np.pi / 360.0)
    print(f"Automatically calculated initial wavelength = {wl_ini:.3f} Å")

    flat_valley_x = int((nickel.peaks_x[1] + nickel.peaks_x[2]) / 2)
    for i in range(len(nickel.x)):
        if nickel.x[i] > flat_valley_x:
            index = i
            break
    flat_valley_y = nickel.y[index]

    nickel.y = nickel.y / flat_valley_y
    nickel.ymax = nickel.ymax / float(flat_valley_y)

    nickel.plot()

    axes = [
        0.5 * flat_valley_x,
        93 * float(wl_ini),
        -0.1 * nickel.ymax,
        1.1 * nickel.ymax,
    ]
    limits = [0.5 * flat_valley_x, 93 * float(wl_ini), 0, 1.1 * nickel.ymax]

    xy_res = [0.81 * flat_valley_x, 0.90 * float(max(nickel.y))]
    xy_ref = [1.60 * flat_valley_x, 0.40 * float(max(nickel.y))]
    xy_fit = [1.85 * flat_valley_x, 0.75 * float(max(nickel.y))]

    fp_area = 0.5 * (nickel.ymax - 1)
    initial = [
        1,
        0.0007 * fp_area,
        0.0,
        float(wl_ini),
        float(zac_ini),
        1.00 * fp_area,
        0.56 * fp_area,
        0.54 * fp_area,
        0.76 * fp_area,
        0.24 * fp_area,
        0.12 * fp_area,
        0.42 * fp_area,
        0.40 * fp_area,
        0.32 * fp_area,
        0.37 * fp_area,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.4,
        0.4,
    ]

    planes = [
        "(111)",
        "(200)",
        "(220)",
        "(311)",
        "(222)",
        "(400)",
        "(331)",
        "(420)",
        "(422)",
        "(333)",
    ]

    refl = reflections_fcc(float(wl_ini), float(zac_ini), lattice=lattice)[:10]
    model = niPeaks10(nickel.x, *initial, lattice=lattice)

    plt.figure(figsize=(9, 6))
    plt.plot(nickel.x, model, "r-", label="Initial model")
    plt.plot(nickel.x, nickel.y, "b-+", label="Ni powder data")
    plt.plot(nickel.x, nickel.y - model, "g-+", label="Residuals")

    plt.legend(loc="best")
    plt.title("Nickel powder diffractogram: Starting point")
    plt.xlabel(r"$2\theta$ (˚)")
    plt.ylabel("Intensity (arb. units)")

    wlength = r"$\lambda = {:.3f}$ Å".format(float(wl_ini))
    zac = r"$2\theta_0 = {:.3f}$˚".format(float(zac_ini))
    plt.text(
        xy_res[0],
        xy_res[1],
        s=wlength + "\n" + zac,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )

    planes_fit = ""
    for i in range(10):
        planes_fit += "{}: {:.1f}˚\n".format(planes[i], float(refl[i]))
    plt.text(
        xy_ref[0],
        xy_ref[1],
        s=planes_fit[:-1],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.axis((axes[0], axes[1], axes[2], axes[3]))
    plt.grid(True)
    plt.tight_layout()
    return axes, limits, xy_res, xy_ref, xy_fit, planes, initial, nickel


def fitting_model_d4nifit(  # noqa: N802 (legacy API)
    nickel: DataXYE,
    limits: list[float],
    initial: list[float],
    lattice: float = NI_LATTICE_PARAMETER,
) -> lm.model.ModelResult:
    """
    Build and fit the Ni powder model for wavelength calibration.

    `lattice` is the Ni lattice parameter in Å (default 3.52 Å). It is held
    fixed: the fitted wavelength scales linearly with it.
    """
    import lmfit as lm

    xfit, yfit, efit = fittingRange(
        limits[0], limits[1], limits[2], limits[3], nickel.x, nickel.y, nickel.e
    )
    _ = efit

    gmodel = lm.Model(niPeaks10)
    gmodel.set_param_hint("I0", vary=True)
    gmodel.set_param_hint("slope", vary=True)
    gmodel.set_param_hint("quad", vary=False)
    gmodel.set_param_hint("wavelength", vary=True)
    gmodel.set_param_hint("twotheta0", vary=True)
    gmodel.set_param_hint("lattice", value=lattice, vary=False)
    for i in range(10):
        gmodel.set_param_hint(f"A{i}", vary=True)
    for i in range(10):
        gmodel.set_param_hint(f"G{i}", vary=False)
    for i in range(10):
        gmodel.set_param_hint(f"S{i}", vary=True)

    result = gmodel.fit(
        yfit,
        x=xfit,
        I0=initial[0],
        slope=initial[1],
        quad=initial[2],
        wavelength=initial[3],
        twotheta0=initial[4],
        A0=initial[5],
        A1=initial[6],
        # Bug fix vs legacy: A2..A4 were initial[6..8] (shifted by one, so
        # initial[9] was never used and A1/A2 started from the same value).
        A2=initial[7],
        A3=initial[8],
        A4=initial[9],
        A5=initial[10],
        A6=initial[11],
        A7=initial[12],
        A8=initial[13],
        A9=initial[14],
        G0=initial[15],
        G1=initial[16],
        G2=initial[17],
        G3=initial[18],
        G4=initial[19],
        G5=initial[20],
        G6=initial[21],
        G7=initial[22],
        G8=initial[23],
        G9=initial[24],
        S0=initial[25],
        S1=initial[26],
        S2=initial[27],
        S3=initial[28],
        S4=initial[29],
        S5=initial[30],
        S6=initial[31],
        S7=initial[32],
        S8=initial[33],
        S9=initial[34],
        lattice=lattice,
    )
    return result


def showing_results_d4nifit(  # noqa: N802 (legacy API)
    result: lm.model.ModelResult,
    nickel: DataXYE,
    axes: list[float],
    limits: list[float],
    xy_res: list[float],
    xy_ref: list[float],
    xy_fit: list[float],
    planes: list[str],
) -> None:
    """
    Print and plot the results of Ni powder calibration fitting (legacy behavior).
    """
    import matplotlib.pyplot as plt

    print()
    print("Results of fitting the nickel powder sample")

    nickel_table: dict[str, list[float]] = {}
    nickel_par: list[float] = []

    for param in result.params.values():
        if param.value != 0:
            relative = (
                abs(float(param.stderr) / float(param.value))
                if param.stderr is not None
                else 0.0
            )
        else:
            relative = 0.0
        nickel_table[param.name] = [
            float(param.value),
            float(param.stderr) if param.stderr is not None else 0.0,
            float(relative),
        ]
        nickel_par.append(float(param.value))
        if param.name == "wavelength":
            wlength = r"$\lambda = ({:.6f} \pm {:.6f})$ Å".format(
                float(param.value), float(param.stderr or 0.0)
            )
        if param.name == "twotheta0":
            zac = r"$2\theta_0 = ({:.6f} \pm {:.6f})$˚".format(
                float(param.value), float(param.stderr or 0.0)
            )

    model = niPeaks10(nickel.x, *nickel_par)  # includes the (fixed) lattice
    lattice = (
        float(result.params["lattice"].value)
        if "lattice" in result.params
        else NI_LATTICE_PARAMETER
    )
    refl = reflections_fcc(nickel_par[3], nickel_par[4], lattice=lattice)[:10]

    def _stderr(name: str) -> float:
        v = nickel_table[name][1]
        return float(v) if v is not None else 0.0

    print(
        f"{result.params['I0'].name:>10} = "
        f"{float(nickel_table['I0'][0]):>9f} ± {_stderr('I0'):>9f}         {float(nickel_table['I0'][2]):>7.3%}"
    )
    print(
        f"{result.params['slope'].name:>10} = "
        f"{float(nickel_table['slope'][0]):>9f} ± {_stderr('slope'):>9f}         {float(nickel_table['slope'][2]):>7.3%}"
    )
    print(
        f"{result.params['quad'].name:>10} = "
        f"{float(nickel_table['quad'][0]):>9f} ± {_stderr('quad'):>9f}         {float(nickel_table['quad'][2]):>7.3%}"
    )
    print(
        f"{result.params['wavelength'].name:>10} = "
        f"({float(nickel_table['wavelength'][0]):>8f} ± {_stderr('wavelength'):>9f}) Å      {float(nickel_table['wavelength'][2]):>7.3%}"
    )
    print(
        f"{result.params['twotheta0'].name:>10} = "
        f"({float(nickel_table['twotheta0'][0]):>8.5f} ± {_stderr('twotheta0'):>9f})˚       {float(nickel_table['twotheta0'][2]):>7.3%}"
    )

    print(103 * "-")
    print(
        f"|{'Plane':>6s} | {'Center':>7s}  | {'Area        ':>20s}  |  {'%   ':>8s} ||"
        f"{'Sigma        ':>19s}  |   {'%   ':>8s} |"
    )
    print(103 * "-")
    for i in range(10):
        ai = f"A{i}"
        si = f"S{i}"
        print(
            f"|{planes[i]:>6s} | {float(refl[i]):>8.5f} | {float(nickel_table[ai][0]):>8.5f} ± {_stderr(ai):>9f}  |  {float(nickel_table[ai][2]):>8.3%} ||"
            f"{float(nickel_table[si][0]):>8.5f} ± {_stderr(si):>9f}  |  {float(nickel_table[si][2]):>9.3%} |"
        )
    print(103 * "-")

    plt.figure(figsize=(9, 6))
    plt.plot(nickel.x, model, "r-", label="Fit")
    plt.plot(nickel.x, nickel.y, "b+", label="Ni powder data")
    plt.plot(nickel.x, nickel.y - model, "g-+", label="Residuals")

    plt.legend(loc="best")
    plt.title("Final fit for " + nickel.filename + ", " + getDate())
    plt.xlabel(r"$2\theta$" + " (˚)")
    plt.ylabel("Intensity (arb. units)")

    fit_report = result.fit_report()
    ite = fit_report.find("function evals")
    chi = fit_report.find("chi-square")
    red = fit_report.find("reduced chi-square")
    aka = fit_report.find("Akaike")
    ite_s = fit_report[ite + 18 : chi - 62]
    chi_s = fit_report[chi + 20 : red - 5]
    red_s = fit_report[red + 20 : aka - 5]

    numors = ""
    if len(nickel.head) > 6 and len(nickel.head[6]) >= 28:
        numors = nickel.head[6][15:28]
    info_fit = "Iterations ={}\nchi-sq ={}\nreduced chi-sq ={}\nNumors: {}".format(
        ite_s, chi_s, red_s, numors
    )

    signature = "d4nifit.ipynb (Dec 2022), G.J. Cuello (ILL), cuello@ill.fr"
    plt.text(
        xy_fit[0],
        xy_fit[1],
        s=info_fit,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )
    plt.text(
        xy_res[0],
        xy_res[1],
        s=wlength + "\n" + zac,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"),
    )
    plt.text(
        axes[0] - 0.05 * (axes[1] - axes[0]),
        axes[2] - 0.1 * (axes[3] - axes[2]),
        s=signature,
        fontsize=6,
    )

    planes_fit = ""
    for i in range(10):
        planes_fit += "{}: {:.1f}˚\n".format(planes[i], float(refl[i]))
    plt.text(
        xy_ref[0],
        xy_ref[1],
        s=planes_fit[:-1],
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.axis((axes[0], axes[1], axes[2], axes[3]))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nickel.basename + ".png")
    print("Results saved as " + nickel.basename + ".png")
    return
