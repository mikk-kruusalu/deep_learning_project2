# Van der Pol equation with Physics Informed Neural Networks

The [van der Pol equation](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator) is a second order ordinary differential equation describing oscillations with non-linear damping.
$$\frac{d^2 x}{dt^2} - \mu(1-x^2) \frac{dx}{dt} + x = 0$$
The equation was derived for electrical circuits that include vacuum tubes. But has found applications in other fields as well such as the Fitzhugh-Nagumo model for action potential propagation in nerve axons.

In PINNs we want the neural network to be the unknown function $NN(t; \theta) \approx x(t)$, where $\theta$ are the parameters. We set up a regular Feed Forward Network and train it with a loss function that converges to zero when the neural network represents the soultion.

In order to have a unique solution we need two initial conditions
$$u(t=0)=u_0 \qquad \frac{d}{dt}u(t=0) = u_0'.$$

We learn the soultion in certain time steps in an interval $t\in[0, t_0]$ for $N$ steps. The loss function consists of two parts -- L$_2$ loss of both of the initial conditions and the L$_2$ loss of the equation
$$
\begin{align*}
\mathcal{L}(\theta) &= (NN(0, \theta) - u_0)^2 + \left( \frac{d}{dt}NN(0, \theta) - u_0' \right) \\
&+ \frac1N \sum_{i=0}^N \left( \frac{d^2}{dt^2}NN(t_i, \theta) - \mu(1-NN(t_i, \theta)^2) \frac{d}{dt}NN(t_i, \theta) + NN(t_i, \theta) \right)
\end{align*}$$
The derivatives can be found with autodifferentiation.