import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm
from cycler import cycler
import matplotlib.animation as animation


#podatki
a = -0.5
b = 1.5
N = 2000
dx = (b-a) / N
Nt = 20000
t = 31.41
dt = t / Nt

sigma_0 = 1/20
k_0 = 50*np.pi
k = 0.2**2
alpha = k**(1/4)
lamb = 0.25
omega = np.sqrt(k)
T_0 = 2*np.pi/omega

t = 100
dt = 2*dx**2


x = np.linspace(a, b, N+1)
x = x[1:-1 ]

def V(x):
	return 0

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
		#T0[i] *= np.sqrt( alpha / np.sqrt(np.pi) ) * np.exp(-alpha**2 * (a+dx*(i+1)-lamb)**2 / 2)
		T0[i] *= (2*np.pi*sigma_0**2)**(-1/4) * np.exp(1j*k_0*(a+dx*(i+1)-lamb)) * np.exp(-(a+dx*(i+1)-lamb)**2/(4*sigma_0**2))
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


def check_distance(sez):
	to_integrate = np.abs(sez)**2*x
	x_ave = simps(to_integrate, dx=dx)
	if x_ave > 0.75:
		return True
	else: return False


def calculate():
	global Nt
	T0 = psi_0_1()
	T = np.array([T0 for _ in range(Nt)])
	A = make_A()
	A_con =np.copy(A).conjugate()
	for i in range(1, Nt):
		T[i] = np.linalg.solve(A, A_con.dot(T[i-1]))
		to_integrate = np.abs(T[i])**2*x
		print('xave', simps(to_integrate, dx=dx))
		if check_distance(T[i]) == True:
			to_integrate = np.abs(T[i])**2*x
			print('xave', simps(to_integrate, dx=dx))
			print('število ciklov', i)
			Nt = i
			print('|||||||||||||||||||||||||||||||||||||')
			break

	return T

def animate(n):
	for i in range(3):
		ax[i].cla()
		ax[i].set_xlim(a, b)
		if i != 2:
			ax[i].set_ylim(-2.8, 2.8)
	ax[0].plot(x, np.real(T[n]), label='t: '+str(format(n/Nt, '.2f')))
	ax[1].plot(x, np.imag(T[n]), label='t: '+str(format(n/Nt, '.2f')))
	ax[2].plot(x, np.abs(T[n])**2, label='t: '+str(format(n/Nt, '.2f')))
	ax[0].plot(x, np.real(exact(x, n*dt/2)), 'r', label='t: '+str(format(n/Nt, '.2f')))
	ax[1].plot(x, np.imag(exact(x, n*dt/2)), 'r', label='t: '+str(format(n/Nt, '.2f')))
	ax[2].plot(x, np.abs(exact(x, n*dt/2))**2, 'r', label='t: '+str(format(n/Nt, '.2f')))
	ax[2].set_xlabel('x')
	ax[0].set_ylabel(r'$\psi_{real}$')
	ax[1].set_ylabel(r'$\psi_{imag}$')
	ax[2].set_ylabel(r'$|\psi|^2$')
	for i in range(3):
		ax[i].legend(loc='upper left')

def animate_one(n, ind):
	ax[ind].set_xlim(a, b)
	ax[ind].plot(x, np.abs(T[n]), label='num')
	ax[ind].plot(x, np.abs(exact(x, n*dt/2)), 'r--', label='analit')
	ax[ind].set_ylabel(r'$|\psi|$')
	ax[ind].legend(loc='upper left')

#set color cycle
def my_cycle(N):
	default_cycler = cycler(color=plt.cm.rainbow(np.linspace(0,1,N)))
	ax[0].set_prop_cycle(default_cycler)
	ax[1].set_prop_cycle(default_cycler)


def exact(x, t):
	konst = (2*np.pi*sigma_0**2)**(-1/4) / np.sqrt(1+1j*t/(2*sigma_0**2))
	ekspo_up = -(x-lamb)**2/(2*sigma_0)**2 + 1j*k_0*(x-lamb) - 1j*k_0**2*t/2
	ekspo_down = 1+1j*t/(2*sigma_0**2)
	return konst * np.exp(ekspo_up/ekspo_down)


from scipy.integrate import simps

def integrator(sez, d):
	return simps(sez, dx=d)

def plot_density():
	fig, ax = plt.subplots()
	t_data = np.linspace(0, 1, Nt)
	d_data = np.zeros(Nt)
	for i in range(Nt):
		d_data[i] = simps(np.abs(T[i])**2, dx=dx)
	ax.plot(t_data, d_data)
	ax.set_ylabel(r'$\rho$')
	ax.set_xlabel(r't')

	print(simps(np.abs(T[0])**2, dx=dx))

def plot_phase():
	fig, ax = plt.subplots(2, 1, sharex=True)
	t_data = np.linspace(0, 1, Nt)
	t_data_ana = np.linspace(0, dt*Nt/2, Nt)
	d_data = np.zeros(Nt)
	d_data_ana = np.zeros(Nt)
	for i in range(Nt):
		to_integrate = np.abs(T[i])**2*x
		d_data[i] = simps(to_integrate, dx=dx)
		to_integrate = np.abs(exact(x, t_data_ana[i]))**2*x
		d_data_ana[i] = simps(to_integrate, dx=dx)
	ax[0].plot(t_data, d_data, 'b')
	ax[0].plot(t_data, d_data_ana, 'r')
	x_plot, y_plot = t_data, calculate_phase(d_data, d_data_ana, t_data)
	ax[1].plot(x_plot, np.array(y_plot))
	ax[1].set_ylabel(r'$\Delta t$')
	ax[0].set_ylabel(r'$\bar{x}$')
	ax[1].set_xlabel(r'$t$')

def calculate_phase(x, y, t_data):
	phase = np.zeros(len(x))
	for i in range(50, len(x)):
		p_y = np.polyfit(t_data[i-50:i], y[i-50:i], 1)
		p_x = np.polyfit(t_data[i-50:i], x[i-50:i], 1)
		k = (p_y[0]+p_x[0])/2
		delta_x = np.abs((y[i-50]-x[i-50])/k)
		print(delta_x)
		phase[i] = delta_x
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
		np.save(name, T)
	return T


fig, ax = plt.subplots(3, 1, sharex=True)

import time

start = time.time()
T = data_save('data2.npy')
end = time.time()
print(end - start)

for i in range(1, len(T)):
	if T[i][1000] == T[0][1000]:
		Nt = i
		break


'''
plot_phase()
plt.show()
'''

plot_density()
plt.show()
#plot
'''
fig, ax = plt.subplots(2, 1, sharex=True)
animate_one(int(round(0)), 0)
ax[0].set_title(r'$t=0$', rotation='vertical', x=1.03, y=0.1)

animate_one(int(round(Nt-1)), 1)
ax[1].set_title(r'$t=1$', rotation='vertical', x=1.03, y=0.1)
ax[1].set_xlabel('x')
plt.show()
'''
#animate


'''
ani = animation.FuncAnimation(fig, animate, frames = range(0, Nt, 60), repeat = True, interval = 50, cache_frame_data=False)
#plt.show()
ani.save('im.gif')
'''