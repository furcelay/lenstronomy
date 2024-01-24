from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe
from lenstronomy.LensModel.Profiles.pie import PIE
from lenstronomy.Util import param_util, util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ["DPIE"]


class DPIE(LensProfileBase):
    """
    class to compute the DUAL PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION (dPIEMD or dPIE)
    based on Eliasdottir (2007) https://arxiv.org/pdf/0710.5636.pdf Appendix A
    and Kassiola & Kovner (1993) https://articles.adsabs.harvard.edu/pdf/1993ApJ...417..450K  Section 4.1.

    Unlike PJaffe_Ellipse, this profile is fully elliptical

    Module name: 'DPIE';

    The 3D density distribution is

    .. math::
        \\rho(r) = \\frac{\\rho_0}{(1+r^2/Ra^2)(1+r^2/Rs^2)}

    with :math:`Rs > Ra`.

    The projected density is

    .. math::
        \\Sigma(R) = \\Sigma_0 \\frac{Ra Rs}{Rs-Ra}\\left(\\frac{1}{\\sqrt{Ra^2+R^2}} - \\frac{1}{\\sqrt{Rs^2+R^2}} \\right)

    with

    .. math::
        \\Sigma_0 = \\pi \\rho_0 \\frac{Ra Rs}{Rs + Ra}

    In the lensing parameterization,

    .. math::
        \\sigma_0 = \\frac{\\Sigma_0}{\\Sigma_{\\rm crit}}

    The asymptotic einstein radius (when :math:`Ra \\to 0` and :math:`Rs \\to \\infty`) is

    .. math::
        \\theta_E = \\sigma_0 \\frac{2 a s^2}{s^2 - a^2}

     The fiducial velocity dispersion :math:`\\sigma_{dPIE}` is

    .. math::
        \\sigma_{dPIE}^2 = \\frac{4}{3} G \\Sigma_0 \\frac{a s^2}{s^2 - a^2}

    which is the same as in Lenstool.
    """

    param_names = ["sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.spherical = PJaffe()
        self.pie = PIE()
        super(DPIE, self).__init__()

    def function(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns double integral of dPIE profile."""
        Ra, Rs = self.spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = sigma0 * Ra * Rs / (Rs - Ra)
        f_a = self.pie._complex_potential(x_, y_, Ra, e)
        f_s = self.pie._complex_potential(x_, y_, Rs, e)
        f_ = (f_a / Ra - f_s / Rs)
        return scale * f_

    def derivatives(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function"""
        Ra, Rs = self.spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = sigma0 * Ra * Rs / (Rs - Ra)
        f_x_a, f_y_a = self.pie._complex_deriv(x_, y_, Ra, e)
        f_x_s, f_y_s = self.pie._complex_deriv(x_, y_, Rs, e)
        f_x, f_y = util.rotate(f_x_a / Ra - f_x_s / Rs,
                               f_y_a / Ra - f_y_s / Rs,
                               -phi)
        return scale * f_x, scale * f_y

    def hessian(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns Hessian matrix of function"""
        Ra, Rs = self.spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = sigma0 * Ra * Rs / (Rs - Ra)
        f_xx_a, f_xy_a, f_yx_a, f_yy_a = self.pie._complex_hessian(x_, y_, Ra, e, q)
        f_xx_s, f_xy_s, f_yx_s, f_yy_s = self.pie._complex_hessian(x_, y_, Rs, e, q)
        f_xx, f_xy, f_yx, f_yy = self.pie._hessian_rotate(f_xx_a / Ra - f_xx_s / Rs,
                                                          f_xy_a / Ra - f_xy_s / Rs,
                                                          f_yx_a / Ra - f_yx_s / Rs,
                                                          f_yy_a / Ra - f_yy_s / Rs,
                                                          -phi)
        return scale * f_xx, scale * f_xy, scale * f_yx, scale * f_yy

    def density_2d(self, x, y, rho0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Projected density.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :param center_x: center of profile
        :param center_y: center of profile
        :return: projected density
        """
        Ra, Rs = self.spherical._sort_ra_rs(Ra, Rs)
        sigma0 = self.rho2sigma(rho0, Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        r_em2 = x_ ** 2 / (1. + e) ** 2 + y_ ** 2 / (1. - e) ** 2
        scale = sigma0 * Ra * Rs / (Rs - Ra)
        sigma = scale * (1 / np.sqrt(Ra**2 + r_em2) - 1 / np.sqrt(Rs**2 + r_em2))
        return sigma

    def mass_3d_lens(self, r, sigma0, Ra, Rs, e1=0, e2=0):
        """

        :param r:
        :param sigma0:
        :param Ra:
        :param Rs:
        :param e1:
        :param e2:
        :return:
        """
        return self.spherical.mass_3d_lens(r, sigma0, Ra, Rs)

    def mass_tot(self, rho0, Ra, Rs):
        """Total mass within the profile.

        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: total mass
        """
        return self.spherical.mass_tot(rho0, Ra, Rs)

    @staticmethod
    def sigma2theta_E(sigma0, Ra, Rs):
        return 2 * Ra * Rs**2 / (Rs**2 - Ra**2) * sigma0

    @staticmethod
    def theta_E2sigma(theta_E, Ra, Rs):
        return (Rs**2 - Ra**2) / (2 * Ra * Rs ** 2) * theta_E

    def sigma2rho(self, sigma0, Ra, Rs):
        return self.spherical.sigma2rho(sigma0, Ra, Rs)

    def rho2sigma(self, rho0, Ra, Rs):
        return self.spherical.rho2sigma(rho0, Ra, Rs)

