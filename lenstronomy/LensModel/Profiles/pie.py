from lenstronomy.Util import param_util, util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ["PIE"]


class PIE(LensProfileBase):
    """
    class to compute the PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION (PIEMD or PIE)
    based on Kassiola & Kovner (1993) https://articles.adsabs.harvard.edu/pdf/1993ApJ...417..450K  Section 4.1.

    This profile is fully elliptical, with ellipticity :math:`\\eps = (1 - q) / (1 + q)`

    Module name: 'PIE';

    The 3D density distribution is

    .. math::
        \\rho(r) = \\frac{\\rho_0}{(1+r^2/Rw^2)}

    The projected density is

    .. math::
        \\Sigma(R) = \\Sigma_0 \\frac{Rw}{\\sqrt{Rw^2 + r_{em}^2}}

    with

    .. math::
        \\Sigma_0 = \\pi Rw \\rho_0

    In the lensing parameterization,

    .. math::
        \\sigma_0 = \\frac{\\Sigma_0}{\\Sigma_{\\rm crit}}

    and

    .. math::
         r_{em} = \\sqrt{\\frac{x^2}{(1 + \\eps)^2} + \\frac{y^2}{(1 - \\eps)^2}}

    The asymptotic einstein radius (when :math:`Rw \\to 0`) is

    .. math::
        \\theta_E = 2 Rw \\sigma_0

     The fiducial velocity dispersion :math:`\\sigma_{dPIE}` is

    .. math::
        \\sigma_{PIE}^2 = \\frac{c^2}{6 \\pi}{D_L}{D_{LS}} \\theta_E

    """

    param_names = ["sigma0", "Rw", "e1", "e2", "center_x", "center_y"]
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
        super(PIE, self).__init__()

    def function(self, x, y, sigma0, Rw, e1=0, e2=0, center_x=0, center_y=0):
        """Spherical lensing potential.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param sigma0: sigma0/sigma_crit (see class documentation above)
        :param Rw: scale radius
        :param e1: ellip1
        :param e2: ellip2
        :param center_x: center of profile
        :param center_y: center of profile
        :return: lensing potential
        """
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        theta_E = self.sigma2theta_E(sigma0, Rw)
        f_ = self._complex_potential(x_, y_, Rw, e)
        return theta_E * f_

    def derivatives(self, x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function"""
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        theta_E = self.sigma2theta_E(sigma0, Rw)
        f_x_, f_y_ = self._complex_deriv(x_, y_, Rw, e)
        f_x, f_y = util.rotate(f_x_, f_y_, -phi)
        return theta_E * f_x, theta_E * f_y

    def hessian(self, x, y, sigma0, Rw, e1, e2, center_x=0, center_y=0):
        """Returns Hessian matrix of function"""
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        theta_E = self.sigma2theta_E(sigma0, Rw)
        f_xx_, f_xy_, f_yx_, f_yy_ = self._complex_hessian(x_, y_, Rw, e, q)
        f_xx, f_xy, f_yx, f_yy = self._hessian_rotate(f_xx_, f_xy_, f_yx_, f_yy_, -phi)
        return theta_E * f_xx, theta_E * f_xy, theta_E * f_yx, theta_E * f_yy

    def density_2d(self, x, y, rho0, Rw, e1, e2, center_x=0, center_y=0):
        """Projected density.

        :param x: projected coordinate on the sky
        :param y: projected coordinate on the sky
        :param rho0: density normalization (see class documentation above)
        :param Rw: scale radius
        :param center_x: center of profile
        :param center_y: center of profile
        :return: projected density
        """
        sigma0 = self.rho2sigma(rho0, Rw)
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        e = param_util.q2e(q)
        x_, y_ = util.rotate(x - center_x, y - center_y, phi)
        r_em2 = x_ ** 2 / (1. + e) ** 2 + y_ ** 2 / (1. - e) ** 2
        theta_E = self.sigma2theta_E(sigma0, Rw)
        scale = theta_E / 2
        sigma = scale * (1 / np.sqrt(Rw**2 + r_em2))
        return sigma

    @staticmethod
    def sigma2theta_E(sigma0, Rw):
        return 2 * Rw * sigma0

    @staticmethod
    def theta_E2sigma(theta_E, Rw):
        return 1 / (2 * Rw) * theta_E

    @staticmethod
    def sigma2rho(sigma0, Rw):
        return 1 / Rw / np.pi * sigma0

    @staticmethod
    def rho2sigma(rho0, Rw):
        return Rw * np.pi * rho0

    def _complex_deriv(self, x, y, Rw, e):
        """
        computes I_w from Eq 4.1.2 in Kassiola & Kovner

        I = prefac * i * log(u / v))
        f_x, f_y = Re(I), Im(I)
        """
        sqe = np.sqrt(e)
        cx = (1. + e) * (1. + e)
        cy = (1. - e) * (1. - e)
        r_em2 = x**2 / (1. + e)**2 + y**2 / (1. - e)**2
        wr_em = np.sqrt(Rw ** 2 + r_em2)
        u = cx * x + (-cy * y + 2 * sqe * wr_em) * 1J
        v = x + (-y + 2 * Rw * sqe) * 1J
        z_log = np.log(u / v)
        prefac = -0.5 * (1. - e**2) / sqe
        return - prefac * np.imag(z_log), prefac * np.real(z_log)
    
    @staticmethod
    def _complex_hessian(x, y, Rw, e, q):
        """
        I = (f_x + f_y * i)
        with I from Eq 4.1.2 in Kassiola & Kovner
        I = prefac * i * log(u / v)) --> I' = prefac * i * (u'/u - v'/v)

        u = cx * x + (-cy * y + 2 * sqe * wr_em) * i
        v = x + (-y + 2 * Rw * sqe) *i

        du_dx = cx + 2 * sqe * x / (cx * wr_em) * i
        du_dy = (-cy + 2 * sqe * y / (cy * wr_em)) * i

        dv_dx = 1
        dv_dy = -i

        f_xx = Re(dI_dx)
        f_xy = f_yx = Re(dI_dy) = Im(dI_dx)
        f_yy = Im(dI_dy)

        simplifying we get to:
        """

        sqe = np.sqrt(e)
        qinv = 1. / q
        cx = (1. + e) * (1. + e)
        cy = (1. - e) * (1. - e)
        prefac = 0.5 * (1. - e ** 2) / sqe
        r_em2 = x ** 2 / cx + y ** 2 / cy
        wr_em = np.sqrt(Rw ** 2 + r_em2)

        u2 = q ** 2 * x ** 2 + (2. * sqe * wr_em - y * qinv) ** 2  # |u|**2
        v_im = 2. * Rw * sqe - y
        v2 = x ** 2 + v_im ** 2  # |v|**2

        f_xx = prefac * (q * (2. * sqe * x ** 2 / cx / wr_em - 2. * sqe * wr_em + y * qinv) / u2 + v_im / v2)
        f_xy = f_yx = prefac * ((2 * sqe * x * y * q / cy / wr_em - x) / u2 + x / v2)
        f_yy = prefac * ((2 * sqe * wr_em * qinv - y * qinv ** 2 - 4 * e * y / cy +
                         2 * sqe * y ** 2 / cy / wr_em * qinv) / u2 - v_im / v2)
        return f_xx, f_xy, f_yx, f_yy

    @staticmethod
    def _complex_potential(x, y, Rw, e):
        sqe = np.sqrt(e)
        prefac = .5 * (1. - e ** 2) / sqe
        cx = (1. + e) * (1. + e)
        cy = (1. - e) * (1. - e)
        r_em2 = x * x / cx + y * y / cy
        e1 = 2. * sqe / (1 - e)
        e2 = 2. * sqe / (1 + e)
        z = np.sqrt(x * x + y * y)
        eta = -.5 * np.asinh(e1 * y / z) + .5 * np.asin(e2 * x / z) * 1J
        zeta = 0.5 * np.log((np.sqrt(r_em2) + np.sqrt(Rw * Rw + r_em2)) / Rw) + 0J
        b1 = np.cosh(eta + zeta)
        b2 = np.cosh(eta - zeta)
        a1 = np.log(np.cosh(eta)**2 / b1 * b2)
        a2 = np.log(b1 / b2)
        c1 = np.sinh(eta * 2.) * a1
        c2 = np.sinh(zeta * 2.) * a2
        ckk = c1 + c2

        return prefac * Rw / np.sqrt(r_em2) * (ckk.imag * x - ckk.real * y)

    @staticmethod
    def _hessian_rotate(f_xx, f_xy, f_yx, f_yy, phi):
        """
         rotation matrix
         R = | cos(t) -sin(t) |
             | sin(t)  cos(t) |

        Hessian matrix
        H = | fxx  fxy |
            | fxy  fyy |

        returns R H R^T

        """
        cos_2phi = np.cos(2 * phi)
        sin_2phi = np.sin(2 * phi)
        a = 1 / 2 * (f_xx + f_yy)
        b = 1 / 2 * (f_xx - f_yy) * cos_2phi
        c = f_xy * sin_2phi
        d = f_xy * cos_2phi
        e = 1 / 2 * (f_xx - f_yy) * sin_2phi
        f_xx_rot = a + b + c
        f_xy_rot = f_yx_rot = d - e
        f_yy_rot = a - b - c
        return f_xx_rot, f_xy_rot, f_yx_rot, f_yy_rot
