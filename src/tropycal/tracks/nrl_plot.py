import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import scipy.ndimage as ndimage
import networkx as nx
from scipy.ndimage import gaussian_filter as gfilt

from ..plot import Plot
from .plot import TrackPlot

#Import tools
from .tools import *
from ..utils import *

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class NRLTrackPlot(TrackPlot):
    def __init__(self):
        super(NRLTrackPlot, self).__init__()

    def plot_storm_nrl(self,forecast,track=None,track_labels='fhr',cone_days=5,domain="dynamic_forecast",ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of the operational NRL forecast track along with observed track data.
        
        Parameters
        ----------
        forecast : dict
            Dict entry containing forecast data.
        track : dict
            Dict entry containing observed track data. Default is none.
        track_labels : str
            Label forecast hours with the following methods:
            '' = no label
            'fhr' = forecast hour
            'valid_utc' = UTC valid time
            'valid_edt' = EDT valid time
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic_forecast" - default. Dynamically focuses the domain on the forecast track.
            "dynamic" - Dynamically focuses the domain on the combined observed and forecast track.
            "lonW/lonE/latS/latN" - Custom plot domain
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Determine if forecast is realtime
        realtime_flag = True if forecast['advisory_num'] == -1 else False
        
        #Set default properties
        default_prop={'dots':True,'fillcolor':'category','linecolor':'k','category_colors':'default','linewidth':1.0,'ms':7.5,'cone_lw':1.0,'cone_alpha':0.6}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
        #--------------------------------------------------------------------------------------
        
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None
        
        #Add storm or multiple storms
        if track != "":
            
            #Check for storm type, then get data for storm
            if isinstance(track, dict) == True:
                storm_data = track
            else:
                raise RuntimeError("Error: track must be of type dict.")
                
            #Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['date']
            
            #Check if there's enough data points to plot
            matching_times = [i for i in sdate if i <= forecast['init']]
            check_length = len(matching_times)
            if check_length >= 2:

                #Subset until time of forecast
                matching_times = [i for i in sdate if i <= forecast['init']]
                plot_idx = sdate.index(matching_times[-1])+1
                lats = storm_data['lat'][:plot_idx]
                lons = storm_data['lon'][:plot_idx]
                vmax = storm_data['vmax'][:plot_idx]
                styp = storm_data['type'][:plot_idx]
                sdate = storm_data['date'][:plot_idx]

                #Account for cases crossing dateline
                if self.proj.proj4_params['lon_0'] == 180.0:
                    new_lons = np.array(lons)
                    new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                    lons = new_lons.tolist()
                
                #Connect to 1st forecast location
                fcst_hr = np.array(forecast['fhr'])
                start_slice = 0
                if 3 in fcst_hr: start_slice = 3
                if realtime_flag == True: start_slice = int(fcst_hr[1])
                iter_hr = np.array(forecast['fhr'])[fcst_hr>=start_slice][0]
                fcst_lon = np.array(forecast['lon'])[fcst_hr>=start_slice][0]
                fcst_lat = np.array(forecast['lat'])[fcst_hr>=start_slice][0]
                fcst_type = np.array(forecast['type'])[fcst_hr>=start_slice][0]
                fcst_vmax = np.array(forecast['vmax'])[fcst_hr>=start_slice][0]
                if fcst_type == "": fcst_type = get_storm_type(fcst_vmax,False)
                if self.proj.proj4_params['lon_0'] == 180.0:
                    if fcst_lon < 0: fcst_lon = fcst_lon + 360.0
                lons.append(fcst_lon)
                lats.append(fcst_lat)
                vmax.append(fcst_vmax)
                styp.append(fcst_type)
                sdate.append(sdate[-1]+timedelta(hours=start_slice))

                #Add to coordinate extrema
                if domain != "dynamic_forecast":
                    if max_lat == None:
                        max_lat = max(lats)
                    else:
                        if max(lats) > max_lat: max_lat = max(lats)
                    if min_lat == None:
                        min_lat = min(lats)
                    else:
                        if min(lats) < min_lat: min_lat = min(lats)
                    if max_lon == None:
                        max_lon = max(lons)
                    else:
                        if max(lons) > max_lon: max_lon = max(lons)
                    if min_lon == None:
                        min_lon = min(lons)
                    else:
                        if min(lons) < min_lon: min_lon = min(lons)
                else:
                    max_lat = lats[-1]+0.2
                    min_lat = lats[-2]-0.2
                    max_lon = lons[-1]+0.2
                    min_lon = lons[-2]-0.2

                #Plot storm line as specified
                if prop['linecolor'] == 'category':
                    type6 = np.array(styp)
                    for i in (np.arange(len(lats[1:]))+1):
                        ltype = 'solid'
                        if type6[i] not in ['SS','SD','TD','TS','HU']: ltype = 'dotted'
                        self.ax.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]],
                                      '-',color=get_colors_sshws(np.nan_to_num(vmax[i])),linewidth=prop['linewidth'],linestyle=ltype,
                                      transform=ccrs.PlateCarree(),
                                      path_effects=[path_effects.Stroke(linewidth=prop['linewidth']*1.25, foreground='k'), path_effects.Normal()])
                else:
                    self.ax.plot(lons,lats,'-',color=prop['linecolor'],linewidth=prop['linewidth'],transform=ccrs.PlateCarree())

                #Plot storm dots as specified
                if prop['dots'] == True:
                    #filter dots to only 6 hour intervals
                    time_hr = np.array([i.strftime('%H%M') for i in sdate])
                    #time_idx = np.where((time_hr == '0300') | (time_hr == '0900') | (time_hr == '1500') | (time_hr == '2100'))
                    lat6 = np.array(lats)#[time_idx]
                    lon6 = np.array(lons)#[time_idx]
                    vmax6 = np.array(vmax)#[time_idx]
                    type6 = np.array(styp)#[time_idx]
                    for i,(ilon,ilat,iwnd,itype) in enumerate(zip(lon6,lat6,vmax6,type6)):
                        mtype = '^'
                        if itype in ['SD','SS']:
                            mtype = 's'
                        elif itype in ['TD','TS','HU']:
                            mtype = 'o'
                        if prop['fillcolor'] == 'category':
                            ncol = get_colors_sshws(np.nan_to_num(iwnd))
                        else:
                            ncol = 'k'
                        self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=0.5,ms=prop['ms'],transform=ccrs.PlateCarree())

        #--------------------------------------------------------------------------------------

        #Error check cone days
        if isinstance(cone_days,int) == False:
            raise TypeError("Error: cone_days must be of type int")
        if cone_days not in [5,4,3,2]:
            raise ValueError("Error: cone_days must be an int between 2 and 5.")
        
        #Error check forecast dict
        if isinstance(forecast, dict) == False:
            raise RuntimeError("Error: Forecast must be of type dict")
            
        #Determine first forecast index
        fcst_hr = np.array(forecast['fhr'])
        start_slice = 0
        if 3 in fcst_hr: start_slice = 3
        if realtime_flag == True: start_slice = int(fcst_hr[1])
        check_duration = fcst_hr[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]

        #Check for sufficiently many hours
        if len(check_duration) > 1:

            #Generate forecast cone for forecast data
            dateline = False
            if self.proj.proj4_params['lon_0'] == 180.0: dateline = True
            cone = self.generate_nrl_cone(forecast,dateline,cone_days)

            #Contour fill cone & account for dateline crossing
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                pass
            else:
                cone_lon = cone['lon']
                cone_lat = cone['lat']
                cone_lon_2d = cone['lon2d']
                cone_lat_2d = cone['lat2d']
                if self.proj.proj4_params['lon_0'] == 180.0:
                    new_lons = np.array(cone_lon_2d)
                    new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                    cone_lon_2d = new_lons.tolist()
                    new_lons = np.array(cone_lon)
                    new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                    cone_lon = new_lons.tolist() 
                cone_2d = cone['cone']
                cone_2d = ndimage.gaussian_filter(cone_2d,sigma=0.5,order=0)
                self.ax.contourf(cone_lon_2d,cone_lat_2d,cone_2d,[0.9,1.1],colors=['#ffffff','#ffffff'],alpha=prop['cone_alpha'],zorder=2,transform=ccrs.PlateCarree())
                self.ax.contour(cone_lon_2d,cone_lat_2d,cone_2d,[0.9],linewidths=prop['cone_lw'],colors=['k'],zorder=3,transform=ccrs.PlateCarree())

            #Plot center line & account for dateline crossing
            center_lon = cone['center_lon']
            center_lat = cone['center_lat']
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(center_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                center_lon = new_lons.tolist()
            self.ax.plot(center_lon,center_lat,color='k',linewidth=2.0,zorder=4,transform=ccrs.PlateCarree())

            #Retrieve forecast dots
            iter_hr = np.array(forecast['fhr'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_lon = np.array(forecast['lon'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_lat = np.array(forecast['lat'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_type = np.array(forecast['type'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_vmax = np.array(forecast['vmax'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            
            #Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(fcst_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                fcst_lon = new_lons.tolist()

            #Plot forecast dots
            for i,(ilon,ilat,itype,iwnd,ihr) in enumerate(zip(fcst_lon,fcst_lat,fcst_type,fcst_vmax,iter_hr)):
                mtype = '^'
                if itype in ['SD','SS']:
                    mtype = 's'
                elif itype in ['TD','TS','HU','']:
                    mtype = 'o'
                if prop['fillcolor'] == 'category':
                    ncol = get_colors_sshws(np.nan_to_num(iwnd))
                else:
                    ncol = 'k'
                #Marker width
                mew = 0.5; use_zorder=5
                if i == 0:
                    mew = 2.0; use_zorder=10
                self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=mew,ms=prop['ms']*1.3,transform=ccrs.PlateCarree(),zorder=use_zorder)

            #Label forecast dots
            if track_labels in ['fhr','valid_utc','valid_edt','fhr_wind_kt','fhr_wind_mph']:
                valid_dates = [forecast['init']+timedelta(hours=int(i)) for i in iter_hr]
                if track_labels == 'fhr':
                    labels = [str(i) for i in iter_hr]
                if track_labels == 'fhr_wind_kt':
                    labels = [f"Hour {iter_hr[i]}\n{fcst_vmax[i]} kt" for i in range(len(iter_hr))]
                if track_labels == 'fhr_wind_mph':
                    labels = [f"Hour {iter_hr[i]}\n{knots_to_mph(fcst_vmax[i])} mph" for i in range(len(iter_hr))]
                if track_labels == 'valid_edt':
                    labels = [str(int(i.strftime('%I'))) + ' ' + i.strftime('%p %a') for i in [j-timedelta(hours=4) for j in valid_dates]]
                    edt_warning = True
                if track_labels == 'valid_utc':
                    labels = [f"{i.strftime('%H UTC')}\n{str(i.month)}/{str(i.day)}" for i in valid_dates]
                self.plot_nhc_labels(self.ax, fcst_lon, fcst_lat, labels, k=1.2)
                
            #Add cone coordinates to coordinate extrema
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                if domain == "dynamic_forecast" or max_lat == None:
                    max_lat = max(center_lat)
                    min_lat = min(center_lat)
                    max_lon = max(center_lon)
                    min_lon = min(center_lon)
                else:
                    if max(center_lat) > max_lat: max_lat = max(center_lat)
                    if min(center_lat) < min_lat: min_lat = min(center_lat)
                    if max(center_lon) > max_lon: max_lon = max(center_lon)
                    if min(center_lon) < min_lon: min_lon = min(center_lon)
            else:
                if domain == "dynamic_forecast" or max_lat == None:
                    max_lat = max(cone_lat)
                    min_lat = min(cone_lat)
                    max_lon = max(cone_lon)
                    min_lon = min(cone_lon)
                else:
                    if max(cone_lat) > max_lat: max_lat = max(cone_lat)
                    if min(cone_lat) < min_lat: min_lat = min(cone_lat)
                    if max(cone_lon) > max_lon: max_lon = max(cone_lon)
                    if min(cone_lon) < min_lon: min_lon = min(cone_lon)

        #--------------------------------------------------------------------------------------

        #Storm-centered plot domain
        if domain == "dynamic" or domain == 'dynamic_forecast':
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
        #--------------------------------------------------------------------------------------
        
        #Identify storm type (subtropical, hurricane, etc)
        first_fcst_wind = np.array(forecast['vmax'])[fcst_hr >= start_slice][0]
        first_fcst_mslp = np.array(forecast['mslp'])[fcst_hr >= start_slice][0]
        first_fcst_type = np.array(forecast['type'])[fcst_hr >= start_slice][0]
        if all_nan(first_fcst_wind) == True:
            storm_type = 'Unknown'
        else:
            subtrop = True if first_fcst_type in ['SD','SS'] else False
            cur_wind = first_fcst_wind + 0
            storm_type = get_storm_classification(np.nan_to_num(cur_wind),subtrop,'north_atlantic')
        
        #Identify storm name (and storm type, if post-tropical or potential TC)
        matching_times = [i for i in storm_data['date'] if i <= forecast['init']]
        if check_length < 2:
            if all_nan(first_fcst_wind) == True:
                storm_name = storm_data['name']
            else:
                storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
                if first_fcst_wind >= 34 and first_fcst_type in ['TD','SD','SS','TS','HU']: storm_name = storm_data['name'];
                if first_fcst_type not in ['TD','SD','SS','TS','HU']: storm_type = 'Storm'
        else:
            storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
            storm_type = 'Storm'
            storm_tropical = False
            if all_nan(vmax) == True:
                storm_type = 'Unknown'
                storm_name = storm_data['name']
            else:
                for i,(iwnd,ityp) in enumerate(zip(vmax,styp)):
                    if ityp in ['SD','SS','TD','TS','HU']:
                        storm_tropical = True
                        subtrop = True if ityp in ['SD','SS'] else False
                        storm_type = get_storm_classification(np.nan_to_num(iwnd),subtrop,'north_atlantic')
                        if np.isnan(iwnd) == True: storm_type = 'Unknown'
                    else:
                        if storm_tropical == True: storm_type = 'Post Tropical Cyclone'
                    if ityp in ['SS','TS','HU']:
                        storm_name = storm_data['name']
        
        #Fix storm types for non-NHC basins
        if 'cone' in forecast.keys():
            storm_type = get_storm_classification(first_fcst_wind,False,forecast['basin'])
        
        #Add left title
        self.ax.set_title(f"{storm_type} {storm_name}",loc='left',fontsize=17,fontweight='bold')

        endash = u"\u2013"
        dot = u"\u2022"
        
        #Get current advisory information
        first_fcst_wind = "N/A" if np.isnan(first_fcst_wind) == True else int(first_fcst_wind)
        first_fcst_mslp = "N/A" if np.isnan(first_fcst_mslp) == True else int(first_fcst_mslp)
        
        #Get time of advisory
        fcst_hr = forecast['fhr']
        start_slice = 0
        if 3 in fcst_hr: start_slice = 1
        if realtime_flag == True: start_slice = 1
        forecast_date = (forecast['init']+timedelta(hours=fcst_hr[start_slice])).strftime("%H%M UTC %d %b %Y")
        forecast_id = forecast['advisory_num']
        
        #Get wind value to display
        if first_fcst_wind == "N/A":
            wind_display_value = "N/A"
        else:
            wind_display_value = knots_to_mph(first_fcst_wind)
        
        if forecast_id == -1:
            title_text = f"Current Intensity: {wind_display_value} mph {dot} {first_fcst_mslp} hPa"
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                title_text += f"\nJTWC Issued: {forecast_date}"
            else:
                title_text += f"\nNHC Issued: {forecast_date}"
        else:
            title_text = f"Current Intensity: {wind_display_value} mph {dot} {first_fcst_mslp} hPa"
            title_text += f"\nForecast Issued: {forecast_date}"
        
        
        #Add right title
        self.ax.set_title(title_text,loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':
            
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            uk = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Unknown', marker='o', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex,sb,uk,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        #Add forecast label warning
        try:
            if edt_warning == True:
                warning_text = "All times displayed are in EDT\n\n"
            else:
                warning_text = ""
        except:
            warning_text = ""
        try:
            warning_text += f"The cone of uncertainty in this product was generated internally\nusing {cone['year']} derived NRL cone radii.\n\n"
        except:
            pass
        
        self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
                transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.savefig(f"{track['id']}_{forecast_date.replace(' ','')}")
            plt.close()
 

    def generate_nrl_cone(self,forecast,dateline,cone_days=5):
        
        r"""
        Generates a cone of uncertainty using forecast data from NHC.
        
        Parameters:
        -----------
        forecast : dict
            Dictionary containing forecast data
        dateline : bool
            If true, grid will be shifted to +0 to +360 degrees longitude. Default is False (-180 to +180 degrees).
        cone_days : int
            Number of forecast days to generate the cone through. Default is 5 days.
        
        """
        print(forecast['fhr'])

        #Determine if forecast is realtime
        realtime_flag = True if forecast['advisory_num'] == -1 else False
        
        #Source: https://www.nhc.noaa.gov/verification/verify3.shtml
        #Radii are in nautical miles
        #cone_climo_hr = [3,12,24,36,48,72,96,120]
        cone_climo_hr = [0,3, 6,12,18,24,30,36,42,48,54,60,66, 72, 78, 84, 90, 96,102,108,114,120]
        cone_size_atl = {}
        cone_size_atl[2020] = [0,6,13,26,33,41,48,55,62,69,77,86,94,103,117,126,135,151,163,173,182,196]
        """
        cone_size_atl[2019] = [16,26,41,54,68,102,151,198]
        cone_size_atl[2018] = [16,26,43,56,74,103,151,198]
        cone_size_atl[2017] = [16,29,45,63,78,107,159,211]
        cone_size_atl[2016] = [16,30,49,66,84,115,165,237]
        cone_size_atl[2015] = [16,32,52,71,90,122,170,225]
        cone_size_atl[2014] = [16,33,52,72,92,125,170,226]
        cone_size_atl[2013] = [16,33,52,72,92,128,177,229]
        cone_size_atl[2012] = [16,36,56,75,95,141,180,236]
        cone_size_atl[2011] = [16,36,59,79,98,144,190,239]
        cone_size_atl[2010] = [16,36,62,85,108,161,220,285]
        cone_size_atl[2009] = [16,36,62,89,111,167,230,302]
        cone_size_atl[2008] = [16,39,67,92,118,170,233,305]
        cone_size_atl[2007] = [16,39,69,98,124,178,253,324]
        cone_size_atl[2006] = [16,42,73,103,131,192,259,335]
        cone_size_atl[2005] = [16,43,77,109,142,207,266,350]
        cone_size_atl[2004] = [16,46,81,117,156,218,275,369]
        cone_size_atl[2003] = [16,47,85,122,162,227,318,433]
        cone_size_atl[2002] = [16,48,85,123,164,233,316,443]
        cone_size_atl[2001] = [16,48,85,124,162,227]
        cone_size_atl[2000] = [16,50,91,132,170,244]
        cone_size_atl[1999] = [16,52,96,136,178,260]
        cone_size_atl[1998] = [16,53,98,141,184,273]
        cone_size_atl[1997] = [16,54,99,144,191,281]
        cone_size_atl[1996] = [16,55,108,155,206,312]
        cone_size_atl[1995] = [16,60,113,166,217,358]
        cone_size_atl[1994] = [16,59,113,168,220,344]
        cone_size_atl[1993] = [16,58,108,162,217,343]
        cone_size_atl[1992] = [16,59,112,166,223,343]
        cone_size_atl[1991] = [16,59,112,167,225,353]
        cone_size_atl[1990] = [16,57,111,168,232,356]
        """

        cone_size_pac = {}
        cone_size_pac[2020] = [0,6,13,25,33,38,45,51,58,65,72,78,85, 91, 96,103,110,115,120,127,133,138]
        """
        cone_size_pac[2019] = [16,25,38,48,62,88,115,145]
        cone_size_pac[2018] = [16,25,39,50,66,94,125,162]
        cone_size_pac[2017] = [16,25,40,51,66,93,116,151]
        cone_size_pac[2016] = [16,27,42,55,70,100,137,172]
        cone_size_pac[2015] = [16,26,42,54,69,100,143,182]
        cone_size_pac[2014] = [16,30,46,62,79,105,154,190]
        cone_size_pac[2013] = [16,30,49,66,82,111,157,197]
        cone_size_pac[2012] = [16,33,52,72,89,121,170,216]
        cone_size_pac[2011] = [16,33,59,79,98,134,187,230]
        cone_size_pac[2010] = [16,36,59,82,102,138,174,220]
        cone_size_pac[2009] = [16,36,59,85,105,148,187,230]
        cone_size_pac[2008] = [16,36,66,92,115,161,210,256]
        """

        #Fix for 2020 that now incorporates 60 hour forecasts
        if forecast['init'].year >= 2020:
            cone_climo_hr = [0,3, 6,12,18,24,30,36,42,48,54,60,66, 72, 78, 84, 90, 96,102,108,114,120]
        if realtime_flag == True: #Realtime
            cone_climo_hr = [forecast['fhr'][1],3,6,12,18,24,30,36,42,48,54,60,66, 72, 78, 84, 90, 96,102,108,114,120]

        #Function for interpolating between 2 times
        def temporal_interpolation(value, orig_times, target_times):
            f = interp.interp1d(orig_times,value)
            ynew = f(target_times)
            return ynew

        #Function for plugging small array into larger array
        def plug_array(small,large,small_coords,large_coords):

            small_lat = np.round(small_coords['lat'],2)
            small_lon = np.round(small_coords['lon'],2)
            large_lat = np.round(large_coords['lat'],2)
            large_lon = np.round(large_coords['lon'],2)

            small_minlat = min(small_lat)
            small_maxlat = max(small_lat)
            small_minlon = min(small_lon)
            small_maxlon = max(small_lon)

            if small_minlat in large_lat:
                minlat = np.where(large_lat==small_minlat)[0][0]
            else:
                minlat = min(large_lat)
            if small_maxlat in large_lat:
                maxlat = np.where(large_lat==small_maxlat)[0][0]
            else:
                maxlat = max(large_lat)
            if small_minlon in large_lon:
                minlon = np.where(large_lon==small_minlon)[0][0]
            else:
                minlon = min(large_lon)
            if small_maxlon in large_lon:
                maxlon = np.where(large_lon==small_maxlon)[0][0]
            else:
                maxlon = max(large_lon)

            large[minlat:maxlat+1,minlon:maxlon+1] = small

            return large

        #Function for finding nearest value in an array
        def findNearest(array,val):
            return array[np.abs(array - val).argmin()]

        #Function for adding a radius surrounding a point
        def add_radius(lats,lons,vlat,vlon,rad):

            #construct new array expanding slightly over rad from lat/lon center
            grid_res = 0.05 #1 degree is approx 111 km
            grid_fac = (rad*4)/111.0

            #Make grid surrounding position coordinate & radius of circle
            nlon = np.arange(findNearest(lons,vlon-grid_fac),findNearest(lons,vlon+grid_fac+grid_res),grid_res)
            nlat = np.arange(findNearest(lats,vlat-grid_fac),findNearest(lats,vlat+grid_fac+grid_res),grid_res)
            lons,lats = np.meshgrid(nlon,nlat)
            return_arr = np.zeros((lons.shape))

            #Calculate distance from vlat/vlon at each gridpoint
            r_earth = 6.371 * 10**6
            dlat = np.subtract(np.radians(lats),np.radians(vlat))
            dlon = np.subtract(np.radians(lons),np.radians(vlon))

            a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats)) * np.cos(np.radians(vlat)) * np.sin(dlon/2) * np.sin(dlon/2)
            c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a));
            dist = (r_earth * c)/1000.0
            dist = dist * 0.621371 #to miles
            dist = dist * 0.868976 #to nautical miles

            #Mask out values less than radius
            return_arr[dist <= rad] = 1

            #Attach small array into larger subset array
            small_coords = {'lat':nlat,'lon':nlon}

            return return_arr, small_coords
        
        #--------------------------------------------------------------------

        #Retrieve cone size for given year
        if forecast['init'].year in cone_size_atl.keys():
            cone_year = forecast['init'].year
            if forecast['basin'] == 'north_atlantic':
                cone_size = cone_size_atl[forecast['init'].year]
            elif forecast['basin'] == 'east_pacific':
                cone_size = cone_size_pac[forecast['init'].year]
            else:
                cone_size = 0
                #raise RuntimeError("Error: No cone information is available for the requested basin.")
        else:
            if forecast['basin'] == 'north_atlantic':
                cone_year = 1990
                cone_size = cone_size_atl[1990]
                msg = f"No cone information is available for the requested year. Defaulting to 1990 cone."
                warnings.warn(msg)
            elif forecast['basin'] == 'east_pacific':
                cone_year = 2008
                cone_size = cone_size_pac[2008]
                msg = f"No cone information is available for the requested year. Defaulting to 2008 cone."
                warnings.warn(msg)
            else:
                cone_year = 2008
                cone_size = 0
                #raise RuntimeError("Error: No cone information is available for the requested basin.")
            #raise RuntimeError("Error: No cone information is available for the requested year.")
        
        #Check if fhr3 is available (or 1st hour for realtime), then get forecast data
        check_fhr = forecast['fhr'][1] if realtime_flag == True else 3
        flag_12 = 0
        if forecast['fhr'][0] == 12:
            flag_12 = 1
            cone_climo_hr = cone_climo_hr[1:]
            fcst_lon = forecast['lon']
            fcst_lat = forecast['lat']
            fhr = forecast['fhr']
            t = np.array(forecast['fhr'])/6.0
            subtract_by = t[0]
            t = t - t[0]
            interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.1) - t[0]
        elif check_fhr in forecast['fhr'] and 1 in forecast['fhr'] and 0 in forecast['fhr']:
            fcst_lon = forecast['lon'][2:]
            fcst_lat = forecast['lat'][2:]
            fhr = forecast['fhr'][2:]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
        elif check_fhr in forecast['fhr'] and 0 in forecast['fhr']:
            idx = np.array([i for i,j in enumerate(forecast['fhr']) if j in cone_climo_hr])
            fcst_lon = np.array(forecast['lon'])[idx]
            fcst_lat = np.array(forecast['lat'])[idx]
            fhr = np.array(forecast['fhr'])[idx]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
        elif forecast['fhr'][1] < 12:
            cone_climo_hr[0] = 0
            fcst_lon = forecast['lon']
            fcst_lat = forecast['lat']
            fhr = forecast['fhr']
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0]/6.0,t[-1]+0.1,0.1)
        else:
            cone_climo_hr[0] = 0
            fcst_lon = forecast['lon']
            fcst_lat = forecast['lat']
            fhr = forecast['fhr']
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.1)

        #Determine index of forecast day cap
        if (cone_days*24) in fhr:
            cone_day_cap = list(fhr).index(cone_days*24)+1
            fcst_lon = fcst_lon[:cone_day_cap]
            fcst_lat = fcst_lat[:cone_day_cap]
            fhr = fhr[:cone_day_cap]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(interp_fhr_idx[0],t[-1]+0.1,0.1)
        else:
            cone_day_cap = len(fhr)
        
        #Account for dateline
        if dateline == True:
            temp_lon = np.array(fcst_lon)
            temp_lon[temp_lon<0] = temp_lon[temp_lon<0]+360.0
            fcst_lon = temp_lon.tolist()

        #Interpolate forecast data temporally and spatially
        interp_kind = 'quadratic'
        if len(t) == 2: interp_kind = 'linear' #Interpolate linearly if only 2 forecast points
        x1 = interp.interp1d(t,fcst_lon,kind=interp_kind)
        y1 = interp.interp1d(t,fcst_lat,kind=interp_kind)
        interp_fhr = interp_fhr_idx * 6
        interp_lon = x1(interp_fhr_idx)
        interp_lat = y1(interp_fhr_idx)
        
        #Return if no cone specified
        if cone_size == 0:
            return_dict = {'center_lon':interp_lon,'center_lat':interp_lat}
            return return_dict

        #Interpolate cone radius temporally
        cone_climo_hr = cone_climo_hr[:cone_day_cap+1]
        cone_size = cone_size[:cone_day_cap+1]
        
        cone_climo_fhrs = np.array(cone_climo_hr)
        if flag_12 == 1:
            interp_fhr += (subtract_by*6.0)
            cone_climo_fhrs = cone_climo_fhrs[1:]
        idxs = np.nonzero(np.in1d(np.array(fhr),np.array(cone_climo_hr)))
        temp_arr = np.array(cone_size)[idxs]
        interp_rad = np.apply_along_axis(lambda n: temporal_interpolation(n,fhr,interp_fhr),axis=0,arr=temp_arr)

        #Initialize 0.05 degree grid
        grid_extent = 9
        if cone_year >= 2004: grid_extent = 8.5
        if cone_year >= 2008: grid_extent = 8
        if cone_year >= 2012: grid_extent = 7.5
        if cone_year >= 2016: grid_extent = 7
        
        gridlats = np.arange(min(interp_lat)-7,max(interp_lat)+7,0.05)
        gridlons = np.arange(min(interp_lon)-7,max(interp_lon)+7,0.05)
        gridlons2d,gridlats2d = np.meshgrid(gridlons,gridlats)

        #Iterate through fhr, calculate cone & add into grid
        large_coords = {'lat':gridlats,'lon':gridlons}
        griddata = np.zeros((gridlats2d.shape))
        for i,(ilat,ilon,irad) in enumerate(zip(interp_lat,interp_lon,interp_rad)):
            temp_grid, small_coords = add_radius(gridlats,gridlons,ilat,ilon,irad)
            plug_grid = np.zeros((griddata.shape))
            plug_grid = plug_array(temp_grid,plug_grid,small_coords,large_coords)
            griddata = np.maximum(griddata,plug_grid)

        return_dict = {'lat':gridlats,'lon':gridlons,'lat2d':gridlats2d,'lon2d':gridlons2d,'cone':griddata,
                       'center_lon':interp_lon,'center_lat':interp_lat,'year':cone_year}
        return return_dict

