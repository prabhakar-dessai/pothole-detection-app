import numpy as np
import cv2
import math
import sys
from queue import Queue


past_lines  = {
    'l':[],
    'r':[]
}
thresh_slope = {
    'l':0,
    'r':0
}

points = np.float32([(40,320),(0,480),(600,320),(640,480)])
transformed_points = np.float32([[0,0],[0,480],[640,0],[640,480]])
M = cv2.getPerspectiveTransform(points,transformed_points)


def update_thresh_slope():
    global past_lines
    global thresh_slope
    for key in ['r','l']:
        temp_slope = 0
        for lines in past_lines[key]:
            temp_slope += slope(lines)
        thresh_slope[key] = temp_slope/len(past_lines[key])
    
            

def slope(line):
    slope_ = 0
    try:
        x1,y1,x2,y2 = line
        slope_ = math.degrees(math.atan2((y2-y1), (x2-x1)))
    except Exception as e:
        pass
    return slope_


def eucliden_distance(line1,line2):
    try:
        x1,y1,x2,y2 = line1
        x3,y3,x4,y4 = line2
        dist1 = point_line_distance(x1, y1, x3, y3, x4, y4)
        dist2 = point_line_distance(x2, y2, x3, y3, x4, y4)
        dist3 = point_line_distance(x3, y3, x1, y1, x2, y2)
        dist4 = point_line_distance(x4, y4, x1, y1, x2, y2)
        return min(dist1, dist2, dist3, dist4)
    except:
        pass


def point_line_distance(x, y, x1, y1, x2, y2):
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return numerator/denominator
   


def lines_roi(lines):
    left, right = [], []
    try:
      for line in lines:
        x1, y1 = int(line[1][0]/line[3]), int(line[1][1]/line[3])
        x2, y2 = int(line[2][0]/line[3]), int(line[2][1]/line[3])
        if abs(y1-y2) > 20 and abs(x1-x2) > 30:
            if (x1 > 0 and x1 <= 140) and (x2 > 0 and x2 <= 140):
                if (y1 >= 300 and y1 < 480) and (y2 >= 300 and y2 < 480):
                    left.append(line)
            if (x1 >= 480 and x1 < 640) and (x2 >= 480 and x2 < 640):
                if (y1 >= 300 and y1 < 480) and (y2 >= 300 and y2 < 480):
                    right.append(line)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
    return [left,right]




def inverse_perspective(image):
    points = np.float32([(40,320),(0,480),(600,320),(640,480)])
    transformed_points = np.float32([[0,0],[0,480],[640,0],[640,480]])
    matrix = cv2.getPerspectiveTransform(points,transformed_points)
    image = cv2.warpPerspective(image,matrix,(640,480))
    return image


def crop_line(pt1,pt2,crop_y):
    try:
        x1,y1 = pt1
        x2,y2 = pt2
        x3 = None
        if x2 != x1 and y2!=y1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            x3 = int((crop_y - b) / m)
        return x3
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)



def get_points(line):
    x1,y1,x2,y2 = line
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points



def get_mapped_lines(h_lines):
    points = np.float32([(40,320),(0,480),(600,320),(640,480)])
    transformed_points = np.float32([[0,0],[0,480],[640,0],[640,480]])
    M = cv2.getPerspectiveTransform(points,transformed_points)
    combined_lines = []
    try:
        for line in h_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            pt1 = (x1,y1)
            pt2 = (x2,y2)
            pt1_mapped = cv2.perspectiveTransform(np.array([[[pt1[0], pt1[1]]]], dtype=np.float32), np.linalg.inv(M))
            pt2_mapped = cv2.perspectiveTransform(np.array([[[pt2[0], pt2[1]]]], dtype=np.float32), np.linalg.inv(M))
            x1,y1 = (int(pt1_mapped[0][0][0]), int(pt1_mapped[0][0][1]))
            x2,y2 = (int(pt2_mapped[0][0][0]), int(pt2_mapped[0][0][1]))
            found = False
            for i, cl in enumerate(combined_lines):
                crho, ctheta = cl[0]
                if abs(rho - crho) < 50 and abs(theta - ctheta) < np.pi/36:
                    combined_lines[i][1][0] += x1
                    combined_lines[i][1][1] += y1
                    combined_lines[i][2][0] += x2
                    combined_lines[i][2][1] += y2
                    combined_lines[i][3] += 1
                    found = True
                    break
            if not found:
                combined_lines.append([line[0], [x1, y1], [x2, y2], 1])
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)

    return combined_lines
  
  
  

def seperate_lines(combined_lines, ylim, check):
    r = []
    l = []
    for line in combined_lines:
        try:
            if line[3] > 1:
                x1, y1 = int(line[1][0]/line[3]), int(line[1][1]/line[3])
                x2, y2 = int(line[2][0]/line[3]), int(line[2][1]/line[3])
                y3 = ylim
                x3 = crop_line((x1,y1),(x2,y2),y3)
                if x3 == None:
                    continue
                slope = 0
                if x2 != x1: 
                    slope = math.degrees(math.atan2((y2-y1), (x2-x1)))
                print()
                if check:
                    if slope > 0 and slope < 60:
                        if x3 > 320:
                            r.append([x3,y3,x2,y2])
                        
                    elif slope < 0 and slope > -70:
                        
                        if x3 < 160:
                            l.append([x1,y1,x3,y3])
                else:
                    if slope > 0:
                        r.append([x3,y3,x2,y2])
                    elif slope < 0:
                        l.append([x1,y1,x3,y3])
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
    return (r,l)


def apply_thresh(image,bright=False):
    rl_lines = []
    global type_
    type_ = 'Adaptive'
    th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    canny = cv2.Canny(th,50,100)
    lines = get_hough(canny)
    rl_lines.append(lines)
    r, l = rl_lines[0]
    return [r, l]


def get_hough(canny):

    h_lines = cv2.HoughLines(canny, rho=1, theta=np.pi/180, threshold=95)
    combined_lines = get_mapped_lines(h_lines)
    return seperate_lines(combined_lines,350,True)


def valid(frame_no, lines):
    global past_lines
    r,l = lines
    if len(past_lines['r']) == 20:
        if len(r) > 0:
            past_lines['r'].pop(0)
            past_lines['r'].append(r[0])
        else:
            past_lines['r'].pop(0)
            past_lines['r'].append(r)
    if len(past_lines['l']) == 20:
        if len(l) > 0:
            past_lines['l'].pop(0)
            past_lines['l'].append(l[0])
        else:
            past_lines['l'].pop(0)
            past_lines['l'].append(l)
        
            
    else:
        if len(r) > 0:
            past_lines['r'].append(r[0])
        else:
            past_lines['r'].append(r)
            
        if len(l) > 0:
            past_lines['l'].append(l[0])
        else:
            past_lines['l'].append(l)
    
    update_thresh_slope()
    
    
    r2,l2 = [],[]
    if len(r) >= 1:
        r = find_leftmost_line(merge_lines(r,5))
        for line in r:
            gradient = slope(line)
            if abs(gradient-thresh_slope['r']) <  5:
                r2.append(line)
    else:
        r2.append([])
        
    if len(l) > 0:
        l = find_rightmost_line(merge_lines(l,5))
        for line in l:
            gradient = slope(line)
            if abs(gradient-thresh_slope['l']) < 5:
                l2.append(line)
    else:
        l2.append([])
    
    return [r2,l2]
        


def merge_lines(lines,threshold_distance):
    merged_lines = []
    merged = [False] * len(lines)

    try:
        for i in range(len(lines)):
            if not merged[i]:
                merged_line = lines[i]

                for j in range(i + 1, len(lines)):
                    if not merged[j]:
                        distance = eucliden_distance(lines[i], lines[j])
                        if distance <= threshold_distance:
                            merged_line = merge_two_lines(merged_line, lines[j])
                            merged[j] = True

                merged_lines.append(merged_line)
    except Exception as e:
        print(e)

    return merged_lines
        
def merge_two_lines(line1,line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    merged_line = [int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)]
    return merged_line        


def find_leftmost_line(lines):
    try:
        leftmost_line = lines[0]
        for line in lines:
            x1, _, _, _ = line
            x1_leftmost, _, _, _ = leftmost_line
            if x1 < x1_leftmost:
                leftmost_line = line
        return [leftmost_line]
    except:
        return [[]]

def find_rightmost_line(lines):
    try:
        rightmost_line = lines[0]
        for line in lines:
            x1, _, _, _ = line
            x1_rightmost, _, _, _ = rightmost_line
            if x1 > x1_rightmost:
                rightmost_line = line
        return [rightmost_line]
    except:
        return [[]]



def lane(frame): 
    frame_no = 0
    lines = preprocess(frame,frame_no)
    return lines


def plot_lines(image,lines):
    h1,w1 = 480,640
    h2,w2,_ = image.shape
    scale_width = w2/w1
    scale_height = h2/h1
    try:
        for lane in lines:
            if len(lane) > 0:
                for line in lane:
                    if len(line) > 0:
                        x1,y1,x2,y2 = line
                        x1 = int(x1 * scale_width)
                        y1 = int(y1 * scale_height)
                        x2 = int(x2 * scale_width)
                        y2 = int(y2 * scale_height)
                        cv2.line(image, (x1,y1), (x2, y2), (255,0,0), 6)
        return image
    except Exception as e:
        print(e)


def preprocess(frame,frame_no):
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5,5),1.5)
    inv_gaus = inverse_perspective(gaus)
    colors = [[255, 0, 0],[ 255, 127, 0],[255, 255, 0],[ 0, 255, 0],[ 0, 0, 255],[46, 43, 95],[139, 0, 255]]
    lines = apply_thresh(inv_gaus,False)
    lines = valid(frame_no,lines)
    return lines
