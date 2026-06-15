import sys
sys.path.insert(0, r"C:\Users\bibhudutta baral\Desktop\pencil-code-master\python")
import pencil as pc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable 


datadir= r"D:\10Gmhd_shock_pcd\data"
param = pc.read.param(datadir=datadir)

dim = pc.read.dim(datadir=datadir)
grid=pc.read.grid(precision='d', datadir=datadir)
x=grid.x[dim.l1:dim.l2+1]
z=grid.z[dim.n1:dim.n2+1]
x_all = grid.x[dim.l1:dim.l2+1]*param.unit_length/1e8
z_all = grid.z[dim.n1:dim.n2+1]*param.unit_length/1e8
var = pc.read.varraw(ivar=55, var_list=['aa', 'uu', 'lnTT', 'lnrho'], datadir=datadir)
T = np.exp(var.lnTT)
rho = np.exp(var.lnrho)
lnTT = var.lnTT[dim.l1:dim.l2+1, dim.m1, dim.n1:dim.n2+1]
log10T = lnTT/2.302585

plt.rcParams.update({
        'font.size': 19,
        'axes.labelweight': 'medium',
        'axes.titleweight': 'medium'
    })

plt.rc('axes', linewidth=1.3)
plt.rc('lines', linewidth=1.7)

cmap = plt.get_cmap('gist_rainbow')
n_levels = 255
c_levels = np.linspace(3.57, 6.00, n_levels)

fig, axs = plt.subplots(figsize=(10, 15))
CF = axs.contourf(x, z, log10T.T, levels=c_levels, cmap=cmap, extend="both") #filled contour plot

axs.set_ylabel(r"$z$ (Mm)")
axs.set_xlabel(r"$x$ (Mm)")
axs.set_aspect('equal')

divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(CF, cax=cax, orientation="vertical")
cbar.set_label(r"$\log_{10} T $")

# Get the min and max from the contour set
tick_min, tick_max = CF.get_clim()
# Define ticks between those
n_ticks = 5
ticks = np.linspace(tick_min, tick_max, n_ticks)
cbar.set_ticks(ticks)
plt.show()
print("===== DATA SUMMARY =====")
print()
print("Simulation time:", var.t)
print("Physical time(s):", var.t*param.unit_time)
print()
print("lnTT shape:", var.lnTT.shape)
print("lnrho shape:", var.lnrho.shape)
print()
print("x range (Mm):", x_all.min(), "to", x_all.max())
print("z range (Mm):", z_all.min(), "to", z_all.max())
print("Temperature range:")
print(T.min(), "to", T.max())
print()
print("Density range:")
print(rho.min(), "to", rho.max())
print()
print("small sample of temperature values:")
print(T[:5, 0,:5])
