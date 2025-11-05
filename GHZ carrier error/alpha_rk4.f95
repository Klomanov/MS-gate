module alpha_rk4
    implicit none

    private
    public :: alpha_tilde_rk4_f

contains

    ! Spline value calc
    real function spline_value(t_bp, oomega_mat, t) result(result)
        real, intent(in) :: t_bp(:)
        real, intent(in) :: oomega_mat(size(t_bp)-1, 4)
        real, intent(in) :: t
        integer :: i
        real :: dt

        do i = 1, size(t_bp)-1
            if (t >= t_bp(i) .and. t <= t_bp(i+1)) then
                dt = t - t_bp(i)
                result = oomega_mat(i,4) + &
                        oomega_mat(i,3) * dt + &
                        oomega_mat(i,2) * dt**2 + &
                        oomega_mat(i,1) * dt**3
                return
            end if
        end do

    end function spline_value


    function f_tilde(t, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat) result(result)
        real, intent(in) :: t
        real, intent(in) :: omegas(:)
        integer, intent(in) :: n_modes
        real, intent(in) :: mu_oomega_psi(n_modes, 3)
        real, intent(in) :: t_bp(:)
        real, intent(in) :: oomega_mat(size(t_bp) - 1, 4)
        complex :: result(size(omegas))

        complex :: sum_f
        real :: omega_t, cos_term, norm_omega_t
        integer :: i, m

        norm_omega_t = spline_value(t_bp, oomega_mat, t)
        do m = 1, size(omegas)
            sum_f = (0.0, 0.0)
            do i = 1, size(mu_oomega_psi)/3
                omega_t = norm_omega_t * mu_oomega_psi(i, 2)
                cos_term = cos(mu_oomega_psi(i, 1) * t + mu_oomega_psi(i, 3))
                sum_f = sum_f + cmplx(cos(omegas(m) * t), sin(omegas(m) * t)) * &
                        omega_t * cos_term
            end do
            result(m) = sum_f
        end do

    end function f_tilde


    function rk4_step(t, alpha, h, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat) result(alpha_new)
        real, intent(in) :: t, h
        real, intent(in) :: omegas(:)
        complex, intent(in) :: alpha(size(omegas))
        integer, intent(in) :: n_modes
        real, intent(in) :: mu_oomega_psi(n_modes, 3)
        real, intent(in) :: t_bp(:)
        real, intent(in) :: oomega_mat(size(t_bp) - 1, 4)
        complex :: alpha_new(size(omegas))

        complex :: k1(size(omegas)), k2(size(omegas)), k3(size(omegas)), k4(size(omegas))

        k1 = h * cmplx(0.0, -1.0) * f_tilde(t, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat)
        k2 = h * cmplx(0.0, -1.0) * f_tilde(t + 0.5 * h, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat)
        k4 = h * cmplx(0.0, -1.0) * f_tilde(t + h, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat)

        alpha_new = alpha + (k1 + 2.0 * k2 + 2.0 * k2 + k4) / 6.0


    end function rk4_step


    subroutine alpha_tilde_rk4_f(t1, t2, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat, n_steps, alpha_result, t_seg)
        real, intent(in) :: t1, t2
        real, intent(in) :: omegas(:)
        real, intent(in) :: mu_oomega_psi(n_modes, 3)
        real, intent(in) :: t_bp(:)
        real, intent(in) :: oomega_mat(size(t_bp) - 1, 4)
        integer, intent(in) :: n_steps, n_modes
        complex, intent(out) :: alpha_result(n_steps, size(omegas))
        real, intent(out) :: t_seg(n_steps)

        real :: dt, t
        complex :: alpha(n_steps, size(omegas))
        integer :: i, j


        t_seg(1) = t1
        do i = 1, n_steps
            do j = 1, size(omegas)
            alpha(i, j) = (0.0, 0.0)
            end do
        end do

        dt = (t2 - t1) / (real(n_steps) - 1)
        do i = 1, n_steps - 1
            alpha(i+1, :) = rk4_step(t_seg(i), alpha(i, :), dt, omegas, n_modes, mu_oomega_psi, t_bp, oomega_mat)
            t_seg(i+1) = t_seg(i) + dt
        end do

        alpha_result = alpha

    end subroutine alpha_tilde_rk4_f

end module alpha_rk4