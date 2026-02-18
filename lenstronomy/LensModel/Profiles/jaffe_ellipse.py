__author__ = "furcelay"

from lenstronomy.LensModel.Profiles.pseudo_jaffe import PseudoJaffe
from lenstronomy.LensModel.Profiles.pseudo_isothermal_ellipse import PseudoIsothermalEllipse
from lenstronomy.Util import param_util, util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ["JaffeEllipse"]


class JaffeEllipse(LensProfileBase):
    """
    class to compute the DUAL PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION (dPIEMD or dPIE)
    based on Eliasdottir (2007) https://arxiv.org/pdf/0710.5636.pdf Appendix A
    and Kassiola & Kovner (1993) https://articles.adsabs.harvard.edu/pdf/1993ApJ...417..450K  Section 4.1.

    Unlike PseudoJaffeEllipsePotential, the mass of this profile is fully elliptical.

    The profile is composed by two PseudoIsothermalEllipse with different scale radius, making a flat core within Ra
    and a steep decay from Rs.

    Module name: 'JAFFE_ELLIPSE';

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
        \\theta_E = 2 Ra \\sigma_0

     The fiducial velocity dispersion :math:`\\sigma_{v}` has different definitions. Here are the conversions
     for Lenstool (lt), Limousin (2005) (lim), and Elíasdóttir (2007) (el):

    .. math::
        \\sigma_{v}^2(lt) = \\frac{c^2}{6 \\pi} \\frac{D_S}{D_{LS}} \\theta_E \\
        \\sigma_{v}^2(lim) = \\frac{c^2}{4 \\pi} \\frac{D_S}{D_{LS}} \\theta_E \\
        \\sigma_{v}^2(lim) = \\frac{c^2}{6 \\pi} \\frac{D_S}{D_{LS}} \\frac{Rs^2 - Ra^2}{Rs^2} \\theta_E

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
    _r_min = 0.0001

    def __init__(self):
        self._spherical = PseudoJaffe()
        self._pie = PseudoIsothermalEllipse()
        super(JaffeEllipse, self).__init__()

    def function(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns double integral of dPIE profile."""
        Ra, Rs = self._spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = np.minimum(np.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        f_a = self._pie._complex_potential(x_, y_, Ra, e)
        f_s = self._pie._complex_potential(x_, y_, Rs, e)
        return scale * (f_a  - f_s)

    def derivatives(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function"""
        Ra, Rs = self._spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = np.minimum(np.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        f_x_a, f_y_a = self._pie._complex_deriv(x_, y_, Ra, e)
        f_x_s, f_y_s = self._pie._complex_deriv(x_, y_, Rs, e)
        f_x, f_y = util.rotate(f_x_a - f_x_s,
                               f_y_a - f_y_s,
                               -phi)
        return scale * f_x, scale * f_y

    def hessian(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns Hessian matrix of function"""
        Ra, Rs = self._spherical._sort_ra_rs(Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = np.minimum(np.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        scale = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        f_xx_a, f_xy_a, f_yx_a, f_yy_a = self._pie._complex_hessian(x_, y_, Ra, e, q)
        f_xx_s, f_xy_s, f_yx_s, f_yy_s = self._pie._complex_hessian(x_, y_, Rs, e, q)
        f_xx, f_xy, f_yx, f_yy = self._pie._hessian_rotate(f_xx_a - f_xx_s,
                                                           f_xy_a - f_xy_s,
                                                           f_yx_a - f_yx_s,
                                                           f_yy_a - f_yy_s,
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
        Ra, Rs = self._spherical._sort_ra_rs(Ra, Rs)
        sigma0 = self.rho2sigma(rho0, Ra, Rs)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = np.minimum(np.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
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
        return self._spherical.mass_3d_lens(r, sigma0, Ra, Rs)

    def mass_tot(self, rho0, Ra, Rs):
        """Total mass within the profile.

        :param rho0: density normalization (see class documentation above)
        :param Ra: core radius
        :param Rs: transition radius from logarithmic slope -2 to -4
        :return: total mass
        """
        return self._spherical.mass_tot(rho0, Ra, Rs)

    @staticmethod
    def sigma2theta_E(sigma0, Ra, Rs):
        return 2 * Ra * sigma0

    @staticmethod
    def theta_E2sigma(theta_E, Ra, Rs):
        return theta_E / (2 * Ra)

    def sigma2rho(self, sigma0, Ra, Rs):
        return self._spherical.sigma2rho(sigma0, Ra, Rs)

    def rho2sigma(self, rho0, Ra, Rs):
        return self._spherical.rho2sigma(rho0, Ra, Rs)

    def _sort_ra_rs(self, Ra, Rs):
        """
        sorts Ra and Rs to make sure Rs > Ra
        """
        Ra = np.where(Ra < Rs, Ra, Rs)
        Rs = np.where(Ra > Rs, Ra, Rs)
        Ra = np.maximum(self._r_min, Ra)
        Rs = np.where(Rs > Ra + self._r_min, Rs, Rs + self._r_min)
        return Ra, Rs