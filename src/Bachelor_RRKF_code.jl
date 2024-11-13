# BUG Integrator.
function DLRA(A, B, r, time_size, step_number)     

    n = size(A,1)

    # Initialisierung.
    U = Matrix{Int}(I, n, r)
    D = zeros(r,r)

    h = time_size/ step_number

    i = 1

    # Runge-Kutta-Verfahren.
    while i <= step_number 
        K = U*D
        
        # K-Schritt
        l_1 = h * (A*K + K*(U'*(A*U))' + B*B'*U)
        l_2 = h * (A*(K+ h/2 * l_1) + (K+ h/2 * l_1)*(U'*(A*U))' + B*B'*U)
        l_3 = h * (A*(K+ h/2 * l_2) + (K+ h/2 * l_2)*(U'*(A*U))' + B*B'*U)
        l_4 = h * (A*(K+ h * l_3) + (K+ h * l_3)*(U'*(A*U))' + B*B'*U)

        K_next  = K + 1/6 * (l_1 + 2*l_2 + 2*l_3 +l_4)

        F = qr!(K_next)
        M = F.Q'*U
        # Update von U.
        U = F.Q
        

        # S-Schritt.
        D = M*D*M'
        
        l_1_hat = h * (U'*(A*(U*D)) + D*(U'*(A*U))' + U'*B*B'*U)
        l_2_hat = h * (U'*(A*(U*(D + h/2 * l_1_hat))) + (D + h/2 * l_1_hat)*(U'*(A*U))' + U'*B*B'*U)
        l_3_hat = h * (U'*(A*(U*(D + h/2 * l_2_hat))) + (D + h/2 * l_2_hat)*(U'*(A*U))' + U'*B*B'*U)
        l_4_hat = h * (U'*(A*(U*(D + h * l_3_hat))) + (D + h * l_3_hat)*(U'*(A*U))' + U'*B*B'*U)

        D_next = D + 1/6 * (l_1_hat + 2*l_2_hat + 2*l_3_hat +l_4_hat)

        # Update von D.
        D = D_next

        i = i+1
    end

    # Berechnung von Q^(1/2) = U*D^(1/2).
    F = svd(D, full = true)
    Q_sqrt = U * F.U*Diagonal(sqrt.(F.S))

    # Es wird angepasst an den Rest des Programmiercodes Q^(1/2) als LeftMatrixSqrt zurückgegeben. 
    return LeftMatrixSqrt(Q_sqrt)
end

# Prädiktionsschritt.
function prediction_rrkf(Phi, mu, Sigma_sqrt, A, B, time_size, step_number, Q_sqrt= nothing)
    n,r = size(Sigma_sqrt)

    #if Q_sqrt == nothing
    #    Q_sqrt = DLRA(A, B, r, time_size, step_number)
    #    F = svd([Phi*Sigma_sqrt Q_sqrt], full = true)
    #else
    #    F = svd([Phi*Sigma_sqrt.factor Q_sqrt.factor], full = true)
    #end

    # Falls Q^(1/2) nicht übergeben wird, wird die Matrix hier berechnet.
    if Q_sqrt == nothing
        Q_sqrt = DLRA(A, B, r, time_size, step_number)
    end

    # Singulärwertzerlegung zur Bestimmung von Pi^(1/2).
    F = svd([Phi*Sigma_sqrt.factor Q_sqrt.factor], full = true)

    U_tilde = F.U[:,1:r]
    D_tilde = Diagonal(F.S[1:r])

    # Bestimmung von dem vorhergesagten mu und dem vorhergesagtem Pi^(1/2).
    mu_minus = Phi*mu
    Pi_sqrt = U_tilde*D_tilde
    
    # Zur Anpassung an den restlichen Programmiercode wird hier LeftMatrixSqrt benutzt.
    return mu_minus, LeftMatrixSqrt(Pi_sqrt)
end

# Korrekturschritt.
function update_rrkf(mu_minus, R_sqrt, C, Pi_sqrt, y, r)
    
    m = size(R_sqrt,1)

    # wenn r < m, dann kann der Korrekturschritt des RRKF ausgeführt werden.
    # Bei r > m oder r = m, muss der Alternativeschritt aus Appendix B benutzt werden.
    if r < m

        R_sqrt_inv = inv(R_sqrt.factor)

        # Bestimmung der Kalman Filter Innovation.
        e = R_sqrt_inv*(y-C*mu_minus)

        # Singulärwertzerlegung
        F = svd((R_sqrt_inv*C*Pi_sqrt.factor)', full=false)
        
        # Berechnung der Kalman-Gain Matrix K.
        D = Diagonal(F.S)
        I_plus_D_inv = inv(I+D*D)
        K = Pi_sqrt.factor*F.U*I_plus_D_inv*D*F.Vt

        # Bestimmung von des korrigierten mu und dem korrigierten Sigma^(1/2).
        mu = mu_minus + K*e
        Sigma_sqrt = Pi_sqrt.factor*F.U*sqrt(I_plus_D_inv)

        # Ermittlung der log-likelihood von y.
        log_of_det = logdet(R_sqrt.factor)
        D_VT_e = D*F.Vt*e

        log_likelihood =  (-m/2*log(2*π) - log_of_det - 1/2 *sum(log(x^2+1) for x in F.S)
        - 1/2* e'*e + 1/2*D_VT_e'*I_plus_D_inv*D_VT_e)

    else
        C_Pi_sqrt = C*Pi_sqrt.factor

        # Bestimmung der Kalman Filter Innovation.
        e = (y-C*mu_minus)

        # Singulärwertzerlegung.
        F = svd([C_Pi_sqrt R_sqrt.factor], full = true)
        
        # Berechnung der Kalman-Gain Matrix K mithilfe K_tilde.
        K_tilde = C_Pi_sqrt'*F.U*Diagonal(1 ./F.S)
        K = Pi_sqrt.factor*K_tilde*Diagonal(1 ./ F.S)*F.U'

        # Singulärwertzerlegung K_tilde.
        FK = svd(K_tilde, full= true)

        # F.S erweitern mit leeren Zeilen, damit die Dimensionen passen.
        d_K_complete = vcat(FK.S, zeros(r - length(FK.S)))

        # Bestimmung von des korrigierten mu und dem korrigierten Sigma^(1/2).
        Sigma_sqrt = Pi_sqrt.factor*FK.U*Diagonal([sqrt(1-diagonal^2) for diagonal in d_K_complete])
        mu = mu_minus + K*e

        # Ermittlung der log-likelihood von y
        D_inv = inv(Diagonal(F.S))
        eT_UD_inv = e'*F.U*D_inv

        log_likelihood = (-m/2*log(2*π) - 1/2 *sum(log(x^2) for x in F.S) 
        - 1/2*eT_UD_inv*eT_UD_inv')

    end

    # Zur Anpassung an den restlichen Programmiercode wird hier LeftMatrixSqrt benutzt.
    return mu, LeftMatrixSqrt(Sigma_sqrt), log_likelihood

end

# Smoothing Algorithm zur Vollständigkeit, wird nicht benutzt.
function backward_kernel_rrkf(Pi_sqrt_next, Phi_next, Sigma_sqrt, mu_next_minus, mu, Q_sqrt_next)

    n,r = size(Sigma_sqrt.factor)

    # Bestimmung des Pseudoinversen von Pi.
    Pi_sqrt_next_pinv= pinv(Pi_sqrt_next.factor)

    # Berechnung von Gamma.
    Gamma = Sigma_sqrt.factor' * Phi_next' * Pi_sqrt_next_pinv

    # Bestimmung der smoothing gain Matrix G.
    G = Sigma_sqrt.factor * Gamma * Pi_sqrt_next_pinv'

    # Ermittlung des shift Vektors v.    
    v = mu - G * mu_next_minus

    # Singulärwertzerlegung.
    F = svd([(I- G*Phi_next)*Sigma_sqrt.factor G*Q_sqrt_next.factor], full=true)
    U_hat =F.U[:,1:r]
    D_hat = Diagonal(F.S[1:r])
    
    # Berechnung des Erwartungswertvektors und der Kovarianzmatrix^(1/2) des backward kernel.
    xi = G*x_next + v
    P_sqrt = U_hat*D_hat 

    # Zur Anpassung an den restlichen Programmiercode wird hier LeftMatrixSqrt benutzt.
    return xi, LeftMatrixSqrt(P_sqrt)
end