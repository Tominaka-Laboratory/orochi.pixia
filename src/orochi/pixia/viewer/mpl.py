import numpy as np
from numpy import amax, amin, arange, meshgrid, ones_like, zeros, ones, nan, where, vstack, append, argmin, abs, pi, sqrt, square, sum, savetxt, array
from matplotlib.backend_bases import key_press_handler

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import path as mpltPath
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import widgets as mw
from matplotlib.widgets import Slider	

# plt.ion()	
# import matplotlib		
# matplotlib.use('Qt5Agg')	

		
#pix		
class Twotheta:
	def __init__(self, fig=None, title="", fignum=1):
		plt.rcParams['font.size']=10
		plt.rcParams['axes.linewidth'] = 1.0
		plt.rcParams['font.family'] = 'Arial'	
		self.labelfont = 10		

		self.fignum = fignum		

		if fig != None:
			self.fig = fig
		else:
			self.fig = plt.figure(num=fignum, figsize=(9,7), dpi=100) 
			
		self.ax = self.fig.add_subplot(111)	
		self.ax.format_coord = lambda x,y: '%.3f, %.3f' % (x,y)				
		
		plt.xlabel('$two theta$ / $\mathrm{\degree}$',fontsize=self.labelfont)
		plt.ylabel('$I$ / a.u.', fontsize=self.labelfont)			
		
		self.ax.set_title(title)			
		self.lines=[]
		self.autoscaling=True

	def plot(self, x, y, number, color="blue", label="", style="-",marker="",markersize="0.5"):
		drawn = len(self.lines)
		x = x.flatten()
		y = y.flatten()
		
		if number > drawn:
			p = self.ax.plot(x, y, label=label, ls=style, color=color, marker=marker,markersize=markersize)
			self.lines += p
		else:
			line = self.lines[number-1]
			line.set_data(x,y)
					
	def connect(self):
		def keys(event):			
			if event.key=="h":	
				print("h:help, a:autoscale")

			elif event.key=="a":
				self.autoscaling = not self.autoscaling
				if self.autoscaling:
					print("autoscaling ON")
				else:
					print("autoscaling OFF")
					
				self.ax.autoscale(self.autoscaling, axis='both', tight=True)
				self.fig.canvas.draw_idle()	

		self.fig.canvas.mpl_connect("key_press_event",keys)		

	def draw(self):
		self.ax.autoscale(enable=self.autoscaling, axis='both', tight=True)		
		self.ax.set_xlim(left=0)
		
		if not plt.fignum_exists(self.fignum):
			self.ax.relim()	
			self.ax.legend(loc='best', prop={'size':12.}, ncol=3, frameon=False, numpoints=3)	
			self.fig.show()		
		
		else:		
			self.ax.relim()		
			self.fig.canvas.draw_idle()	
						
	def close(self):
		plt.close(self.fignum)
						
class Viewer:
	def __init__(self, calculator=None, fig=None, title="", fignum=1):
		self.labelfont = 10
		plt.rcParams['font.size']=10
		plt.rcParams['axes.linewidth'] = 1.0
		plt.rcParams['font.family'] = 'Arial'	

			
		self.calculator = calculator
		self.fignum = fignum
		self.title = title
		
		if not plt.fignum_exists(self.fignum):
			if fig != None:
				self.fig = fig
			else:
				self.fig = plt.figure(num=self.fignum, figsize=(6,6))		
			
		else:
			plt.clf()		
			self.fig = plt.gcf()

		self.shown = False
		self.mask_connected = False
		self.image_connected =False
		
		self.ax = self.fig.add_subplot(111)	
		divider = make_axes_locatable(self.ax)
		self.cax = divider.append_axes("right",size="3%",pad=0.05)
		self.autoscaling=False	
		
		self.fig.canvas.draw()		
		self.bb = self.fig.canvas.copy_from_bbox(self.ax.bbox)	
		self.scroll = 0
		
	def draw(self):
		self.ax.autoscale(False)	
		# self.fig.canvas.set_window_title(self.title)		

		self.ax.set_title(self.title)
		plt.show()

	def imread(self, image, vmin=0.):				
		self.image = image	
										
		if not "im" in dir(self):
			self.im = self.ax.imshow(self.image, vmin=vmin, vmax=amax(self.image)*1.05, origin="lower", interpolation='nearest')	
			self.amp = amax(self.image)
			self.fig.colorbar(self.im, cax=self.cax)	
		else:
			self.im.set_data(self.image)		
			
	def connect_image(self):	
		if self.image_connected:
			pass
			
		self.imagex = None
		self.imagey = None		
		self.lineplot = False	
	
		def keyboard_handler(event):
			if event.key=="b":	
				self.brighter()

			elif event.key=="d":	
				self.darker()
					
			elif event.key=="h":	
				print("h:help, a:autoscale, b:brighter, d:darker")

			elif event.key=="a":
				self.autoscaling = not self.autoscaling
				if self.autoscaling:
					print("autoscaling ON")
				else:
					print("autoscaling OFF")
				self.ax.autoscale(self.autoscaling)	
				
			elif event.key =="up":
				self.scroll = 1
				
			elif event.key =="down":
				self.scroll = -1				
								
		def monitor(event):
			scale = 10.
			color1 = "white"
			color2 = "magenta"
			
			if event.button==3 and self.lineplot:
				readx = event.xdata
				ready = event.ydata
				y = self.image[int(ready),:]
				x = self.image[:,int(readx)]
				center = self.image[int(ready),int(readx)]
				
				self.profile_x.set_data([0, self.calculator["pixx"]], [ready,ready])
				self.profile_y.set_data([readx,readx],[0, self.calculator["pixy"]])
				
				self.anot.set_position((int(readx), int(ready)))
				self.anot.set_text(str(int(center)))				

				self.xline.set_data(arange(0,self.calculator["pixx"]),  y/float(amax(y)) * float(self.calculator["pixy"])/scale + float(self.calculator["pixy"])/scale*(scale*0.85))
				self.yline.set_data(x/float(amax(x))*float(self.calculator["pixx"])/scale,arange(0,self.calculator["pixy"]))
				
				self.fig.canvas.draw_idle()
				
			elif event.button==1 and event.dblclick==1 and self.lineplot:
				self.profile_x.remove()
				self.profile_y.remove()
				
				self.anot.remove()					
									
				self.xline.remove()
				self.yline.remove()
				self.lineplot = False
				self.fig.canvas.draw_idle()
			
			elif event.button==1 and event.dblclick==1 and self.lineplot == False:
				readx = self.calculator["x0"]
				ready = self.calculator["y0"]
				
				y = self.image[int(ready),:]
				x = self.image[:,int(readx)]
				center = self.image[int(ready), int(readx)]
				
				self.profile_x, = self.ax.plot([0,self.calculator["pixx"]], [self.calculator["y0"], self.calculator["y0"]], color=color2,lw=0.3)
				self.profile_y, = self.ax.plot([self.calculator["x0"], self.calculator["x0"]], [0,self.calculator["pixy"]], color=color2,lw=0.3)
				self.anot = self.ax.annotate(str(int(center)), xy=(int(readx), int(ready)), color='white')		
				
				self.xline, = self.ax.plot(arange(0,self.calculator["pixx"]),  y/float(amax(y))*float(self.calculator["pixy"])/scale  +  float(self.calculator["pixy"])/scale*(scale*0.85), color=color1,lw=0.3)
				self.yline, = self.ax.plot(x/float(amax(x))*float(self.calculator["pixx"])/scale,arange(0,self.calculator["pixy"]), color=color1,lw=0.3)		
				
				self.lineplot = True
				self.fig.canvas.draw_idle()		
			
		self.fig.canvas.mpl_connect("key_press_event", keyboard_handler)
		self.fig.canvas.mpl_connect("button_press_event", monitor)	
		
		self.image_connected = True
			
	def rings(self, tt, angles, rings=10):
		#x = arange(tt.shape[1])
		#y = arange(tt.shape[0])	
		#X,Y  = meshgrid(x,y)
			
		cont = self.ax.contour(tt, levels=angles[:rings], alpha=0.5, colors=['white'], linewidths=0.8, linestyles="dashed", origin = "lower")
		cont.clabel(fmt='%1.1f', fontsize=8)	

		
	def initialize_mask(self):
		self.tempmask = ones_like(self.image)
		self.tempmask[:,:] = nan		
				
		if "mask" in dir(self.calculator):
			self.mask = self.calculator.mask			
		else:
			self.mask = self.calculator.get_mask()
		self.mask[where(self.mask != 0)] = nan			
		self.im_temp = self.ax.imshow(self.tempmask, vmin=-1,vmax = self.amp/10., origin="lower",interpolation='nearest',cmap='rainbow')			
		self.im_mask = self.ax.imshow(self.mask, vmin=-1, origin="lower", interpolation='nearest', cmap='rainbow')	

	def connect_mask(self):
		if self.mask_connected:
			pass
		
		self.status = 1
		self.keeps = 1
		self.modes = 1
		self.polygon =[]
		
		sx, sy = self.calculator["pixx"], self.calculator["pixy"]
		x, y = meshgrid(arange(sx), arange(sy))
		x, y = x.flatten(), y.flatten()
		points = vstack((x,y)).T			
	
		def button_press(event):			
			if event.button==1 and self.status==1:
				try:
					self.left_x = int(event.xdata)
					self.left_y = int(event.ydata)
					print(("x=" + str(self.left_x)))
					print(("y=" + str(self.left_y)))		
					
					if self.modes==1:
						self.status=2
						print("point 1 (rectangle)")
						
					elif self.modes==2:
						self.status=2
						self.polygon += [(self.left_x, self.left_y)]
						print("point 1 (polygon)")
				except:pass
				
			elif event.button==1 and self.status ==2:
				self.right_x = int(event.xdata)
				self.right_y = int(event.ydata)
				print(("x="+str(self.right_x)))
				print(("y="+str(self.right_y)))				
				
				if self.modes ==1:
					self.status = 3
					print("point 2 (rectangle)")

					xlist = [self.left_x, self.right_x]
					ylist = [self.left_y, self.right_y]
					x1 = amin(xlist)
					x2 = amax(xlist)
					y1 = amin(ylist)
					y2 = amax(ylist)	
					 
					self.tempmask[y1:y2, x1:x2] = True
					
				elif self.modes == 2:
					self.polygon += [(self.right_x, self.right_y)]					
					
					path = mpltPath.Path(self.polygon)
					inside = path.contains_points(points, radius=1e-9)
					grid = inside.reshape((sy,sx))				
					self.tempmask[grid] = True

				self.im_temp.set_data(self.image * self.tempmask)
				self.fig.canvas.draw_idle()
				
			elif event.button==1 and event.dblclick==1 and self.status==3:
				print("accepted")
				self.status=1
				
				if self.modes==1:
					xlist = [self.left_x, self.right_x]
					ylist = [self.left_y, self.right_y]
					x1 = amin(xlist)
					x2 = amax(xlist)
					y1 = amin(ylist)
					y2 = amax(ylist)		
					
					self.calculator["mask_rectangle"] = np.append(self.calculator["mask_rectangle"], [y1,y2,x1,x2]).reshape(-1,4)
					
				elif self.modes==2:
					self.calculator["mask_polygon"] = list(self.calculator["mask_polygon"]) + [self.polygon]
					self.polygon = []
					
				self.update_mask()
				self.draw()
		
			elif event.button==3:  
				print("canceled")
				self.tempmask[:,:] = nan
				self.polygon =[]
				self.im_temp.set_data(self.tempmask)
				self.fig.canvas.draw_idle()
				self.status=1
							
		def key_press(event):
			if event.key=="a" and self.status>0:
				self.keeps = self.status
				self.status = 0
				print("deactivated")
				
			elif event.key=="a" and self.status==0:
				self.status = self.keeps
				print("activated")
				
			elif event.key=="i":	
				print("initialized")
				self.calculator["mask_polygon"] = []
				self.calculator["mask_rectangle"] = []
				self.status=1		
				self.initialize_mask()
				self.update_mask()
				self.fig.canvas.draw_idle()					

			elif event.key=="b":	
				print("brighter")
				self.brighter()

			elif event.key=="d":	
				print("darker")
				self.darker()
				
			elif event.key=="h":	
				print("Mouse Left Click: Select (two points), /nMouse Left DClick: Approve, /nMouse Right Click: Cancel, /nKey i: Initialize, /nKey a: On/Off, /nKey p: polygonal mode, /nr: accept, /nKey h: Help")

			elif event.key=="p":	
				if self.modes == 1:
					print("polygonal draw")
					self.modes = 2
				elif self.modes == 2:
					print("rectangular draw")
					self.modes = 1
					
			elif event.key=="r":	
				print("accepted")
				self.status=1
				
				if self.modes==1:
					xlist = [self.left_x, self.right_x]
					ylist = [self.left_y, self.right_y]
					x1 = amin(xlist)
					x2 = amax(xlist)
					y1 = amin(ylist)
					y2 = amax(ylist)					
					self.calculator["mask_rectangle"] = np.append(self.calculator["mask_rectangle"], [y1,y2,x1,x2]).reshape(-1,4)
					
				elif self.modes==2:
					self.calculator["mask_polygon"] = list(self.calculator["mask_polygon"]) + [self.polygon]	
					
				self.tempmask[:,:] = nan
				self.polygon =[]					
				self.update_mask()
				self.draw()

		self.fig.canvas.mpl_connect("button_press_event", button_press)
		self.fig.canvas.mpl_connect("key_press_event", key_press)	
		
		self.mask_connected = True
						
	def update_mask(self):	
		if not "im_temp" in dir(self):
			self.im_temp = self.ax.imshow(self.tempmask, vmin=-1,vmax = self.amp/10., origin="lower",interpolation='nearest',cmap='rainbow')			
			self.im_mask = self.ax.imshow(self.mask, vmin=-1, origin="lower", interpolation='nearest', cmap='rainbow')		
		else:
			self.im_temp.set_data(self.tempmask)
			self.im_mask.set_data(self.mask)	
		
		
	def brighter(self):
		self.amp /= 4.
		self.im.set_clim(vmax = self.amp)
		self.fig.canvas.draw_idle()	
		
	def darker(self):
		self.amp *= 2.
		self.im.set_clim(vmax = self.amp)
		self.fig.canvas.draw_idle()	
		
								
	def close(self):
		plt.close(self.fignum)	
		
class Map:
	def __init__(self, xlabel='twotheta /deg', ylabel='time /sec'):
		self.labelfont = 10
		plt.rcParams['font.size']=10
		plt.rcParams['axes.linewidth'] = 1.0
		plt.rcParams['font.family'] = 'Arial'		
		

	def pcolor(self, data, xaxis=[], yaxis=[]):
		def coord2boundary(coordinates):		
			diff = coordinates[1:]-coordinates[:-1]
			mid = coordinates[:-1]+diff/2.
			init = coordinates[0]-diff[0]/2.
			last = coordinates[-1]+diff[-1]/2.
			boundary = append(init, mid)
			boundary = append(boundary, last)
			return boundary
			
		self.xaxis = xaxis
		self.yaxis = yaxis
		xb = coord2boundary(xaxis)	
		yb = coord2boundary(yaxis)				
			
		if not plt.fignum_exists(4):
			self.fig = plt.figure(num=4, figsize=(6,6))
			self.im_ax = self.fig.add_subplot(111)	
			self.im_ax.set_xticklabels([])
			self.im_ax.set_yticklabels([])
			
			divider = make_axes_locatable(self.im_ax)
			self.x_ax = divider.append_axes("top",size="20%",pad=0.05)
			self.y_ax = divider.append_axes("right",size="10%",pad=0.05)
			self.c_ax = divider.append_axes("right",size="3%",pad=0.10)
			self.b_ax = divider.append_axes("bottom",size="40%",pad=0.15)
			
			#self.x_ax.set_yticklabels([])
			#self.x_ax.set_xticklabels([])
			#self.x_ax.set_xlim(min,max)
			self.x_p, = self.x_ax.plot([self.xaxis[0], self.xaxis[-1]], [0,0], color="blue", lw=0.8)		
			self.y_p, = self.y_ax.plot([0,0], [self.yaxis[0], self.yaxis[-1]], color="blue", lw=0.8)	

			p4a, = ax4.plot(0,0,marker='',ls='-',color="red",label="I0",lw=0.5)
			p4b, = ax4.plot(0,0,marker='',ls='-',color="blue",label="I",lw=0.5)
			plt.legend(loc='upper right',prop={'size':8.},ncol=2,frameon=False,numpoints=3,handletextpad=0.5)#bbox_to_anchor=(0.5, 1.0)			

			self.p1, = self.im_ax.pcolorfast([xb, yb], data, cmap="plasma")
			self.im_ax.autoscale(True,axis='y',tight=True)	
			self.p1.set_clim()
			
			cbar = plt.colorbar(self.p1, cax=self.c_ax)
			cl = plt.getp(cbar.ax, 'ymajorticklabels')
			plt.setp(cl,family='Arial',fontsize=fsize)		
			
			def select(event):
				if event.button==3:
					readx = event.xdata
					ready = event.ydata
					
					rx = argmin(abs(self.xaxis - readx) )
					ry = argmin(abs(self.yaxis - ready) )
					
					self.x_p.set_data( self.xaxis, data[rx,:] )
					self.x_p.autoscale(True, axis='y', tight=True)	
					self.x_p.set_title("y = " + str(ready), size =10)	
					
					self.y_p.set_data( data[:,ry], self.yaxis )
					self.y_p.autoscale(True, axis='x', tight=True)
					self.y_p.set_title("x = " + str(readx), size =10)						

					self.fig.canvas.draw_idle()
						
			self.fig.canvas.mpl_connect("button_press_event",select)			
		
		else:
			self.p1.set_data([xb, yb], data)
			self.p1.set_clim()
			self.ax.relim()	
			self.ax.autoscale(axis='y')
			self.fig.canvas.draw_idle()		




