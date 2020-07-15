def calculate_distance(color_intrin,x,y):
    color_intrin = self.color_intrin
    ix,iy = self.ix, self.iy
    udist = self.depth_frame.get_distance(ix,iy)
    vdist = self.depth_frame.get_distance(x, y)
    #print udist,vdist

    point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
    point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
    #print str(point1)+str(point2)

    dist = math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
            point1[2] - point2[2], 2))
    #print 'distance: '+ str(dist)
    return dist