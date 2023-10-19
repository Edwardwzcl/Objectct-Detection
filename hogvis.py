import matplotlib.pyplot as plt
import numpy as np
import skimage.draw

def hogvis(descriptor,bsize=8,norient=9):
    """
    This function visualizes a hog descriptor using little
    icons to indicate orientation. We follow the convention
    that the first orientation bin starts at a gradient 
    orientations of -pi/2 and the last bin ends at pi/2
    
    Parameters
    ----------
    descriptor : 3D float array 
         HOG descriptor values

    bsize : int
        The size of the spatial bins in pixels, defaults to 8
        
    norient : int
        The number of orientation histogram bins, defaults to 9
    
    Returns
    -------
    hog_image : 2D float array
        Visualization of the hog descriptor with oriented line segments.
        
    """   

    d_h = descriptor.shape[0]
    d_w = descriptor.shape[1]
    hog_image = np.zeros((d_h*bsize,d_w*bsize), dtype=float)
    
    #radius of a spatial bin
    radius = bsize // 2
    orient = np.arange(norient)

    #angle of bin mid-points 0..pi
    orient_angle = (np.pi * (orient + .5) / norient)
    
    #end points of a line at each orientation
    vr = -(radius-0.5) * np.cos(orient_angle)
    vc = (radius-0.5) * np.sin(orient_angle)

    for r in range(d_h):
        for c in range(d_w):
            for o, dr, dc in zip(orient,vr,vc):
                centre = tuple([r*bsize + radius, c*bsize + radius])
                rr, cc = skimage.draw.line(int(centre[0] - dc), int(centre[1] + dr),
                                          int(centre[0] + dc), int(centre[1] - dr))
                hog_image[rr,cc] += descriptor[r, c, o]

    return hog_image




def test_hogvis():
    # test code
    # create a 3x3 descriptor where each bin only has a single orientation present
    a = np.array([[[1.,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]],
         [[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0]],
         [[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]])

    #normalize descriptor
    s = np.sum(a,axis=2)
    for i in range(9):
        a[:,:,i] = a[:,:,i] / s

    #visualize
    hogim = hogvis(a,bsize=20)
    plt.imshow(hogim)
    plt.show()
