import os
import sys
import re
import glob
import math
import csv
import argparse
import subprocess
import urllib.request
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import (
    SkyCoord, ICRS, Galactic, FK4, FK5,
    Angle, Latitude, Longitude
)
from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.convolution import convolve_fft, Gaussian2DKernel

from reproject import reproject_interp
from radio_beam import Beam



def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print(x)
def check_url_exists(url):
    try:
        # Send a HEAD request to the URL (faster than a full GET request)
        response = requests.head(url, allow_redirects=True, timeout=5)
        
        # Check if the status code is in the range 200-299 (successful responses)
        if response.status_code // 100 == 2:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        # If any exception occurs (e.g., connection error, timeout), the URL doesn't exist
        print(f"Error: {e}")
        return False

def vlass_image(VLASS_id,tile_id,RA,DEC):
 URL = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/'
 r = requests.head(URL)
 if r.status_code == 200:
    print('url was not found')
 else:
    VLASS_id='VLASS2.1'
    URL = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/'
 r = requests.head(URL)
 if r.status_code == 200:
    print('url was not found')
 else:
    VLASS_id='VLASS2.2'
    URL = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/'   
 try:
  print(URL)
  urlpath = urlopen(URL)
  string = (urlpath.read().decode('utf-8')).split("\n")
  vals = np.array([val.strip() for val in string])
  ra1=[];ra2=[];ra3=[];dec1=[];dec2=[];dec3=[];strapnd=[]
  for line in vals:
               if 'ql.'+tile_id in line:
                JName_regex = 'J\d{6}[+|-]\d{6}[.]\d\d[.]\d{4}\S{3}'
                m = re.search(JName_regex, line).group(0)
                #print(m)
                ra1.append(str(m[1:3]))
                ra2.append(str(m[3:5]))
                ra3.append(str(m[5:7]))
                dec1.append(str(m[7:10]))
                dec2.append(str(m[10:12]))
                dec3.append(str(m[12:14]))
                strapnd.append(m[14:])
  separation=[];deg_cor=[]
  for i in range(len(ra1)): 
             c1 = SkyCoord(str(int(ra1[i]))+' '+str(int(ra2[i]))+' '+str(int(ra3[i]))+' '+str(int(dec1[i]))+' '+str(int(dec2[i]))+' '+str(int(dec3[i])), unit=(u.hourangle, u.deg),frame='fk4')
             RA2=c1.ra.value
             DEC2=c1.dec.value
             deg_cor.append(c1) 
             c2 = SkyCoord(RA*u.deg, DEC*u.deg, frame='fk4') 
             sep = c1.separation(c2).deg
             separation.append(sep)
  min_deg=min(separation)
  min_deg_pos=separation.index(min(separation))
#found_JName = m.group(0)
  found_JName=str((ra1[min_deg_pos]))+str((ra2[min_deg_pos]))+str((ra3[min_deg_pos]))+str((dec1[min_deg_pos]))+str((dec2[min_deg_pos]))+str((dec3[min_deg_pos]))+strapnd[min_deg_pos]
  full_directory_name = VLASS_id.replace('v2','') + '.ql.' + tile_id + '.' +'J'+ found_JName
  return full_directory_name,URL,RA2,DEC2,min_deg,VLASS_id
 except:
   pass  


def vlass_ms(VLASS_id,tile_id):
  URL_New = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/' + full_directory_name + '/casa_pipescript.py'
  try:
    url = URL_New  #read casa_pipescript.py
    search_file_for_ms =  urllib.request.urlopen(url)
    for line in search_file_for_ms:
     decoded_line = line.decode("utf-8")
     if decoded_line.find(str(VLASS_id)) != -1:
       start_value = int(decoded_line.find(str(VLASS_id)))
       end_value = int(decoded_line.find('.ms'))
       measurement_set_name = decoded_line[start_value:end_value]
       measurement_set_list.append(measurement_set_name)
    return measurement_set_list
  except:
     pass
 




Object_name = sys.argv[1]
ra=sys.argv[2]
dec=sys.argv[3]
Im_Size=1

with open('Tile_Boundaries.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

   
Im_Size_Degrees = Im_Size/3600.
RA=float(ra);DEC=float(dec)   
RA_Right = RA + Im_Size_Degrees
RA_Left = RA - Im_Size_Degrees
Dec_Up = DEC + Im_Size_Degrees
Dec_Down = DEC - Im_Size_Degrees

measurement_set_list = []

for tile in data:
        Dec_Tile_Start = float(tile[1])
        Dec_Tile_End = float(tile[2])
        Dec_Tile_Center = (Dec_Tile_End - Dec_Tile_Start)/2 + Dec_Tile_Start
        RA_Tile_Start = float(tile[3])*15
        RA_Tile_End = float(tile[4])*15
        RA_Tile_Center = (RA_Tile_End - RA_Tile_End)/2 + RA_Tile_Start        
        RA_L = [RA_Left, RA_Right,RA_Left,RA_Right, RA, RA, RA, RA_Left, RA_Right]
        DEC_L = [Dec_Up, Dec_Down, Dec_Down, Dec_Up, DEC, Dec_Up, Dec_Down, DEC, DEC]
        for i in range(0,len(RA_L)):
            if RA_L[i] > RA_Tile_Start and RA_L[i] < RA_Tile_End and DEC_L[i] > Dec_Tile_Start and DEC_L[i] < Dec_Tile_End:
                tile_id = tile[0]
                VLASS_id = tile[5]
                URL = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/'              
                page = requests.get(URL).text
                JName_regex = 'J\d{6}[+]\d{6}[.]\d\d[.]\d{4}\S{3}'
                m = re.search(JName_regex, page)
                if m:
                    found_JName = m.group(0)
                    full_directory_name = VLASS_id + '.ql.' + tile_id + '.' + found_JName
                    URL_New = "https://archive-new.nrao.edu/vlass/quicklook/" + VLASS_id + '/' + tile_id + '/' + full_directory_name + '/casa_pipescript.py'                  
                    try:
                        url = URL_New  #read casa_pipescript.py
                        search_file_for_ms =  urllib.request.urlopen(url)
                        for line in search_file_for_ms:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.find(str(VLASS_id)) != -1:
                                start_value = int(decoded_line.find(str(VLASS_id)))
                                end_value = int(decoded_line.find('.ms'))
                                measurement_set_name = decoded_line[start_value:end_value]
                                measurement_set_list.append(measurement_set_name)
                    except:
                        pass
     
                  
print("The unique measurement sets required for cluster "+str(Object_name)+" are:")                    
unique(measurement_set_list)
#print(URL) 
#print('\n')
try:
   directory_name=vlass_image(VLASS_id,tile_id,RA,DEC)
  #print( directory_name)
   full_directory_name=directory_name[0]
   URL_name=directory_name[1];RA2=directory_name[2];DEC2=directory_name[3];min_deg=directory_name[4]
   vlsid=directory_name[5]
   if vlsid=='VLASS1.1v2':
     vlsid2='VLASS2.1';vlsid3='VLASS3.1'
     full_directory_name2=full_directory_name.replace("1.1","2.1")
     full_directory_name3=full_directory_name.replace("1.1","3.1")
   if vlsid=='VLASS1.2v2':
     vlsid2='VLASS2.2';vlsid3='VLASS3.2'
     full_directory_name2=full_directory_name.replace("1.2","2.2")
     full_directory_name3=full_directory_name.replace("1.2","3.2")
   if vlsid=='VLASS2.1':
     vlsid2='VLASS1.1v2';vlsid3='VLASS3.1'
     full_directory_name2=full_directory_name.replace("2.1","1.1")
     full_directory_name3=full_directory_name.replace("2.1","3.1")
   if vlsid=='VLASS2.2':
     vlsid2='VLASS1.2v2';vlsid3='VLASS3.2'
     full_directory_name2=full_directory_name.replace("2.2","1.2")
     full_directory_name3=full_directory_name.replace("1.2","3.2")
  
   mslist=vlass_ms(vlsid,tile_id)

   url = 'https://archive-new.nrao.edu/vlass/quicklook/' + vlsid + '/' + tile_id + '/' + full_directory_name +'/'+full_directory_name+'.I.iter1.image.pbcor.tt0.subim.fits'
   if check_url_exists(url):
      print(vlsid,tile_id,full_directory_name)
      #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid + '/' + tile_id + '/' + full_directory_name +'/'+full_directory_name+'.I.iter1.image.pbcor.tt0.subim.fits')
      #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid + '/' + tile_id + '/' + full_directory_name + '/casa_commands.log')
      #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid +'_casa_commands.log')
   else:
      full_directory_name=full_directory_name.replace(".v1",".v2")
      print(vlsid,tile_id,full_directory_name)
      #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid + '/' + tile_id + '/' + full_directory_name +'/'+full_directory_name+'.I.iter1.image.pbcor.tt0.subim.fits')
      #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid + '/' + tile_id + '/' + full_directory_name + '/casa_commands.log')
      #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid +'_casa_commands.log')
 

   url = 'https://archive-new.nrao.edu/vlass/quicklook/' + vlsid2 + '/' + tile_id + '/' + full_directory_name2 +'/'+full_directory_name2+'.I.iter1.image.pbcor.tt0.subim.fits'

   if check_url_exists(url):
     print(vlsid,tile_id,full_directory_name2)
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid2 + '/' + tile_id + '/' + full_directory_name2 +'/'+full_directory_name2+'.I.iter1.image.pbcor.tt0.subim.fits')
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid2 + '/' + tile_id + '/' + full_directory_name2 + '/casa_commands.log')
     #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid2 +'_casa_commands.log')
   else:
     full_directory_name2=full_directory_name2.replace(".v1",".v2")
     print(vlsid,tile_id,full_directory_name2)
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid2 + '/' + tile_id + '/' + full_directory_name2 +'/'+full_directory_name2+'.I.iter1.image.pbcor.tt0.subim.fits')
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid2 + '/' + tile_id + '/' + full_directory_name2 + '/casa_commands.log')
     #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid2 +'_casa_commands.log')

   url= 'https://archive-new.nrao.edu/vlass/quicklook/' + vlsid3 + '/' + tile_id + '/' + full_directory_name3 +'/'+full_directory_name3+'.I.iter1.image.pbcor.tt0.subim.fits'

   if check_url_exists(url):
     print(vlsid,tile_id,full_directory_name3)
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid3 + '/' + tile_id + '/' + full_directory_name3 +'/'+full_directory_name3+'.I.iter1.image.pbcor.tt0.subim.fits')
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid3 + '/' + tile_id + '/' + full_directory_name3 + '/casa_commands.log')
     #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid3 +'_casa_commands.log')
   else:
     full_directory_name3=full_directory_name3.replace(".v1",".v2")
     print(vlsid,tile_id,full_directory_name3)
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid3 + '/' + tile_id + '/' + full_directory_name3 +'/'+full_directory_name3+'.I.iter1.image.pbcor.tt0.subim.fits')
     #os.system('wget https://archive-new.nrao.edu/vlass/quicklook/' + vlsid3 + '/' + tile_id + '/' + full_directory_name3 + '/casa_commands.log')
     #os.system('mv'+' '+'casa_commands.log'+' '+Object_name+'_'+ vlsid3 +'_casa_commands.log')
   
   #os.system('mv'+' '+full_directory_name+'.I.iter1.image.pbcor.tt0.subim.fits'+' '+Object_name+'_VLASS_'+ str(vlsid)+'_image.fits')
   #os.system('mv'+' '+full_directory_name2+'.I.iter1.image.pbcor.tt0.subim.fits'+' '+Object_name+'_VLASS_'+ str(vlsid2)+'_image.fits')
   #os.system('mv'+' '+full_directory_name3+'.I.iter1.image.pbcor.tt0.subim.fits'+' '+Object_name+'_VLASS_'+ str(vlsid3)+'_image.fits')
   print(Object_name,RA,DEC,RA2,DEC2,min_deg,vlsid,tile_id,full_directory_name,URL_name,mslist[0])
except:
   pass
print('\n')  






