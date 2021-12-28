import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm
from cycler import cycler
import matplotlib.animation as animation


#podatki
a = -20
b = 20
N = 300
dx = (b-a) / N
Nt = 200000
t = 31.41
dt = t / Nt

k = 0.2**2
alpha = k**(1/4)
lamb = 10
omega = np.sqrt(k)
T_0 = 2*np.pi/omega

N_T_0 = 21
t = N_T_0*2*T_0
dt = t / Nt


x = np.linspace(a, b, N+1)
x = x[1:-1 ]

def V(x):
	return k*x**2 / 2

#matrika
def make_A():
	b = 1j * dt / 2 / dx**2
	p1 = np.full(N-2, 0.j+1.)
	for i in range(1, N-1):
		p1[i-1] *= -b/2 
	p0 = np.full(N-1, 0.j+1.) # srednja diagonala
	for i in range(1, N):
		p0[i-1] += 1 + b + 1j * dt / 2 * V(a + dx*i)

	return np.diag(p0) + np.diag(p1,k=-1) + np.diag(p1,k=1)

#initial conditions
def psi_0_1():
	T0 = np.full(N-1, 0j+1)
	for i in range(0, N-1):
		T0[i] *= np.sqrt( alpha / np.sqrt(np.pi) ) * np.exp(-alpha**2 * (a+dx*(i+1)-lamb)**2 / 2)

	return T0

def solve():
	T0 = psi_0_1()
	T = np.array([T0 for _ in range(Nt)])
	ax[0].plot(x, np.real(T[0]), label=0)
	ax[1].plot(x, np.imag(T[0]), label=0)
	ax[2].plot(x, np.abs(T[0])**2)
	A = make_A()
	A_con =np.copy(A).conjugate()

	for i in range(1, Nt):
		T[i] = np.linalg.solve(A, A_con.dot(T[i-1]))
		ax[0].plot(x, np.real(T[i]), label=i)
		ax[1].plot(x, np.imag(T[i]), label=i)
		ax[2].plot(x, np.abs(T[i])**2)

def calculate():
	T0 = psi_0_1()
	T = np.array([T0 for _ in range(Nt)])
	A = make_A()
	A_con =np.copy(A).conjugate()
	for i in range(1, Nt):
		T[i] = np.linalg.solve(A, A_con.dot(T[i-1]))

	return T

def animate(n):
	for i in range(3):
		ax[i].cla()
		ax[i].set_xlim(a, b)
		ax[i].set_ylim(-0.8, 0.8)
	ax[0].plot(x, np.real(T[n]), label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ (num)')
	ax[1].plot(x, np.imag(T[n]), label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ (num)')
	ax[2].plot(x, np.abs(T[n])**2, label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ (num)')
	ax[0].plot(x, np.real(exact(x, n*dt/2)), 'r', label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ ')
	ax[1].plot(x, np.imag(exact(x, n*dt/2)), 'r', label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ ')
	ax[2].plot(x, np.abs(exact(x, n*dt/2))**2, 'r', label='t: '+str(format(n*t/Nt/T_0/2, '.2f'))+r' $T_0$ ')
	ax[2].set_ylim(-0.5, 0.5)
	ax[2].set_xlabel('x')
	ax[0].set_ylabel(r'$\psi_{real}$')
	ax[1].set_ylabel(r'$\psi_{imag}$')
	ax[2].set_ylabel(r'$|\psi|^2$')
	for i in range(3):
		ax[i].legend(loc='upper left')

def animate_one(n, ind):
	ax[ind].set_xlim(a/2, b)
	ax[ind].plot(x, np.real(T[n]), label='real num')
	ax[ind].plot(x, np.imag(T[n]), label='imag num')
	ax[ind].plot(x, np.real(exact(x, n*dt/2)), 'r--', label='real')
	ax[ind].plot(x, np.imag(exact(x, n*dt/2)), 'g--', label='imag')
	ax[ind].set_ylabel(r'$\psi$')
	ax[ind].legend(loc='upper left')

#set color cycle
def my_cycle(N):
	default_cycler = cycler(color=plt.cm.rainbow(np.linspace(0,1,N)))
	ax[0].set_prop_cycle(default_cycler)
	ax[1].set_prop_cycle(default_cycler)


def exact(x, t):
	konst = np.sqrt(alpha/np.sqrt(np.pi))
	ksi = alpha*x
	ksi_lamb = alpha*lamb
	real_part = -0.5 * (ksi-ksi_lamb*np.cos(omega*t))**2
	imag_part = omega*t/2 + ksi*ksi_lamb*np.sin(omega*t) - ksi_lamb**2/4*np.sin(2*omega*t)
	return konst * np.exp(real_part - 1j*imag_part)

from scipy.integrate import simps

def integrator(sez, d):
	return simps(sez, dx=d)

def plot_density():
	fig, ax = plt.subplots()
	t_data = np.linspace(0, t, Nt//100)
	d_data = np.zeros(Nt//100)
	for i in range(Nt//100):
		d_data[i] = simps(np.abs(T[i*100])**2, dx=dx)
	ax.plot(t_data, d_data)
	ax.set_xticks([t/N_T_0*i for i in range(N_T_0)])
	ax.set_xticklabels([i for i in range(N_T_0)])
	ax.set_ylabel(r'$\rho$')
	ax.set_xlabel(r't [$\mathrm{T}_0$]')

	print(simps(np.abs(T[0])**2, dx=dx))

def plot_phase():
	fig, ax = plt.subplots(2, 1, sharex=True)
	t_data = np.linspace(0, t/2, Nt//100)
	t_data_ana = np.linspace(0, t/4, Nt//100)
	d_data = np.zeros(Nt//100)
	d_data_ana = np.zeros(Nt//100)
	for i in range(Nt//100):
		to_integrate = np.abs(T[i*100//2])**2*x
		d_data[i] = simps(to_integrate, dx=dx)
		to_integrate = np.abs(exact(x, t_data_ana[i]))**2*x
		d_data_ana[i] = simps(to_integrate, dx=dx)
	ax[0].plot(t_data, d_data)
	ax[0].plot(t_data, d_data_ana, 'r')
	x_plot, y_plot = find_min(t_data, calculate_phase(d_data, d_data_ana, t_data))
	ax[1].plot(x_plot, np.array(y_plot)*2*np.pi/(Nt/N_T_0)*100)
	ax[1].set_xticks([t/N_T_0*i for i in range(N_T_0//2+1)])
	ax[1].set_xticklabels([i for i in range(N_T_0//2+1)])
	ax[1].set_ylabel(r'$\delta$')
	ax[0].set_ylabel(r'$\bar{x}$')
	ax[1].set_xlabel(r't [$\mathrm{T}_0$]')

def calculate_phase(x, y, t_data):
	phase = np.zeros(len(x))
	for i in range(20, len(x)):
		if check_mono(x[i-20:i]) == True and check_mono(y[i-20:i]) == True:
			print('lin')
			p_x = np.polyfit(t_data[i-20:i], x[i-20:i], 1)
			x_x = -p_x[1]/p_x[0]
			print('x ', x_x)
			p_y = np.polyfit(t_data[i-20:i], y[i-20:i], 1)
			x_y = -p_y[1]/p_y[0]
			print('y ', x_y)
		else:
			print('quad')
			p_x = np.polyfit(t_data[i-20:i], x[i-20:i], 2)
			x_x = -p_x[1]/p_x[0]/2
			print('x ', x_x)
			p_y = np.polyfit(t_data[i-20:i], y[i-20:i], 2)
			x_y = -p_y[1]/p_y[0]/2
			print('y ', x_y)
		if np.abs(phase[i-1] - np.abs(x_y-x_x)) < 30:
			phase[i] = np.abs(x_y-x_x)
		else:
			phase[i] = phase[i-1]
	return phase

def find_min(t, phase):
	t_new = [t[0]]
	phase_new = [phase[0]]
	for i in range(1, len(phase)-1):
		if phase[i-1] > phase[i] and phase[i+1] > phase[i]:
			t_new.append(t[i])
			phase_new.append(phase[i])
	print(t_new, phase_new)
	return t_new, phase_new

def check_mono(sez):
	checker = bool(sez[0] > sez[1])
	for i in range(len(sez)-1):
		if bool(sez[i] > sez[i+1]) != checker:
			return False
	return True

def data_save(name):
	try:
		T = np.load(name)
	except:
		print('exception!!!')
		T = calculate()
		np.save('data1', T)
	return T


fig, ax = plt.subplots(3, 1, sharex=True)

import time

start = time.time()
T = data_save('data1.npy')
end = time.time()
print(end - start)

plot_phase()
plt.show()

#plot
'''
animate_one(int(round(Nt*T_0/t*2*1)), 0)
ax[0].set_title(r'$\mathrm{T}_0$', rotation='vertical', x=1.03, y=0.1)

animate_one(int(round(Nt*T_0/t*2*5)), 1)
ax[1].set_title(r'$5\mathrm{T}_0$', rotation='vertical', x=1.03, y=0.1)

animate_one(int(round(Nt*T_0/t*2*10)), 2)
ax[2].set_title(r'$10\mathrm{T}_0$', rotation='vertical', x=1.03, y=0.1)
ax[2].set_xlabel('x')
plt.show()
'''
#animate


'''
ani = animation.FuncAnimation(fig, animate, frames = range(0, Nt, 50), repeat = True, interval = 100, cache_frame_data=False)
#plt.show()
ani.save('im.gif')
'''