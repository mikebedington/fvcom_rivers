from math import sqrt, pi, sin, cos, tan, atan2 as arctan2
import csv


def OStoLL(E,N):

    #E, N are the British national grid coordinates - eastings and northings
    a, b = 6377563.396, 6356256.909     #The Airy 180 semi-major and semi-minor axes used for OSGB36 (m)
    F0 = 0.9996012717                   #scale factor on the central meridian
    lat0 = 49*pi/180                    #Latitude of true origin (radians)
    lon0 = -2*pi/180                    #Longtitude of true origin and central meridian (radians)
    N0, E0 = -100000, 400000            #Northing & easting of true origin (m)
    e2 = 1 - (b*b)/(a*a)                #eccentricity squared
    n = (a-b)/(a+b)

    #Initialise the iterative variables
    lat,M = lat0, 0

    while N-N0-M >= 0.00001: #Accurate to 0.01mm
        lat = (N-N0-M)/(a*F0) + lat;
        M1 = (1 + n + (5./4)*n**2 + (5./4)*n**3) * (lat-lat0)
        M2 = (3*n + 3*n**2 + (21./8)*n**3) * sin(lat-lat0) * cos(lat+lat0)
        M3 = ((15./8)*n**2 + (15./8)*n**3) * sin(2*(lat-lat0)) * cos(2*(lat+lat0))
        M4 = (35./24)*n**3 * sin(3*(lat-lat0)) * cos(3*(lat+lat0))
        #meridional arc
        M = b * F0 * (M1 - M2 + M3 - M4)          

    #transverse radius of curvature
    nu = a*F0/sqrt(1-e2*sin(lat)**2)

    #meridional radius of curvature
    rho = a*F0*(1-e2)*(1-e2*sin(lat)**2)**(-1.5)
    eta2 = nu/rho-1

    secLat = 1./cos(lat)
    VII = tan(lat)/(2*rho*nu)
    VIII = tan(lat)/(24*rho*nu**3)*(5+3*tan(lat)**2+eta2-9*tan(lat)**2*eta2)
    IX = tan(lat)/(720*rho*nu**5)*(61+90*tan(lat)**2+45*tan(lat)**4)
    X = secLat/nu
    XI = secLat/(6*nu**3)*(nu/rho+2*tan(lat)**2)
    XII = secLat/(120*nu**5)*(5+28*tan(lat)**2+24*tan(lat)**4)
    XIIA = secLat/(5040*nu**7)*(61+662*tan(lat)**2+1320*tan(lat)**4+720*tan(lat)**6)
    dE = E-E0

    #These are on the wrong ellipsoid currently: Airy1830. (Denoted by _1)
    lat_1 = lat - VII*dE**2 + VIII*dE**4 - IX*dE**6
    lon_1 = lon0 + X*dE - XI*dE**3 + XII*dE**5 - XIIA*dE**7

    #Want to convert to the GRS80 ellipsoid. 
    #First convert to cartesian from spherical polar coordinates
    H = 0 #Third spherical coord. 
    x_1 = (nu/F0 + H)*cos(lat_1)*cos(lon_1)
    y_1 = (nu/F0+ H)*cos(lat_1)*sin(lon_1)
    z_1 = ((1-e2)*nu/F0 +H)*sin(lat_1)

    #Perform Helmut transform (to go between Airy 1830 (_1) and GRS80 (_2))
    s = -20.4894*10**-6 #The scale factor -1
    tx, ty, tz = 446.448, -125.157, + 542.060 #The translations along x,y,z axes respectively
    rxs,rys,rzs = 0.1502,  0.2470,  0.8421  #The rotations along x,y,z respectively, in seconds
    rx, ry, rz = rxs*pi/(180*3600.), rys*pi/(180*3600.), rzs*pi/(180*3600.) #In radians
    x_2 = tx + (1+s)*x_1 + (-rz)*y_1 + (ry)*z_1
    y_2 = ty + (rz)*x_1  + (1+s)*y_1 + (-rx)*z_1
    z_2 = tz + (-ry)*x_1 + (rx)*y_1 +  (1+s)*z_1

    #Back to spherical polar coordinates from cartesian
    #Need some of the characteristics of the new ellipsoid    
    a_2, b_2 =6378137.000, 6356752.3141 #The GSR80 semi-major and semi-minor axes used for WGS84(m)
    e2_2 = 1- (b_2*b_2)/(a_2*a_2)   #The eccentricity of the GRS80 ellipsoid
    p = sqrt(x_2**2 + y_2**2)

    #Lat is obtained by an iterative proceedure:   
    lat = arctan2(z_2,(p*(1-e2_2))) #Initial value
    latold = 2*pi
    while abs(lat - latold)>10**-16: 
        lat, latold = latold, lat
        nu_2 = a_2/sqrt(1-e2_2*sin(latold)**2)
        lat = arctan2(z_2+e2_2*nu_2*sin(latold), p)

    #Lon and height are then pretty easy
    lon = arctan2(y_2,x_2)
    H = p/cos(lat) - nu_2

#Uncomment this line if you want to print the results
    #print [(lat-lat_1)*180/pi, (lon - lon_1)*180/pi]

    #Convert to degrees
    lat = lat*180/pi
    lon = lon*180/pi

    #Job's a good'n. 
    return lat, lon


def read_csv_unheaded(filename, cols):

    output = []

    for i in range(0,cols):
        this_list = []

        with open(filename, 'rt') as this_file:
            this_file_data = csv.reader(this_file)

            for row in this_file_data:
                this_list.append(row[i])

        output.append(this_list)
    return output

def OS_text_convert(grid_ref_str):

    grid_sqr = grid_ref_str[0:2]
    ref = grid_ref_str[2:].strip()
    
    try:
        ref_x = ref[0:int(len(ref)/2)]
        ref_y = ref[int(len(ref)/2):]

    except:
        print('Grid reference x and y different lengths')
        return        

    ref_grid = OS_letter_conversion_list() 

    try:
        x_add = [ref_grid[2][x] for x, y in enumerate(ref_grid[0]) if y == grid_sqr][0]
        y_add = [ref_grid[1][x] for x, y in enumerate(ref_grid[0]) if y == grid_sqr][0]
    
    except:
        print('Grid 2-letter identifier not recognised')
        return

    if x_add == '0':
        x_add = ''

    if y_add == '0':
        y_add = ''

    new_grid = [float(x_add + ref_x) , float(y_add + ref_y)]
    
    return new_grid


def OS_letter_conversion_list():
    osl_list = [
    ['SV','SW','SX','SY','SZ','TV','SR','SS','ST','SU','TQ','TR','SM','SN','SO','SP','TL','TM','SH','SJ','SK','TF','TG','SC','SD','SE','TA','NW','NX','NY','NZ','OV','NR','NS','NT','NU','NL','NM','NN','NO','NF','NG','NH','NJ','NK','NA','NB','NC','ND','HW','HX','HY','HZ','HU','HT','HP'],
    ['0','0','0','0','0','0','1','1','1','1','1','1','2','2','2','2','2','2','3','3','3','3','3','4','4','4','4','5','5','5','5','5','6','6','6','6','7','7','7','7','8','8','8','8','8','9','9','9','9','10','10','10','10','11','11','12'],
    ['0','1','2','3','4','5','1','2','3','4','5','6','1','2','3','4','5','6','2','3','4','5','6','2','3','4','5','1','2','3','4','5','1','2','3','4','0','1','2','3','0','1','2','3','4','0','1','2','3','1','2','3','4','3','4','4']]
    return osl_list

    
