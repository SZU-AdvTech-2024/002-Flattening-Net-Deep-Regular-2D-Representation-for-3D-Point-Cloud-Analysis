import math
import numpy as np
import PIL.Image as pil
import matplotlib.pylab  as plt


def plot_pcd_multi_rows(filename, pcds, titles=None, suptitle='', sizes=None, cmap='Greys', zdir='y',
                         xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), zlim=(-0.6, 0.6)):
    if sizes is None:
        sizes = [0.2 for i in range(pcds.shape[0])]

    if titles is None:
        titles = np.arange(pcds.shape[0])

    rows = 1
    total = pcds.shape[0]
    while(total % rows != 0):
        total +=1
    cols = total // rows

    # pcds = normalize_point_cloud(pcds)
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(rows):
        elev = 135
        azim = -65
        for j, pcd in enumerate(pcds[i*cols:(i+1)*cols]):
            color = np.zeros(pcd.shape[0])
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=sizes[i*cols + j], cmap=cmap, vmin=-1, vmax=0.5)
            # ax.set_title(titles[i*cols + j])
            #ax.text(0, 0, titles[i][j], color="green")
            ax.set_axis_off()
            #ax.set_xlabel(titles[i][j])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    #plt.xticks(np.arange(len(pcds)), titles[:len(pcds)])

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def normalize_point_cloud(inputs):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    #print("shape",input.shape)
    C = inputs.shape[-1]
    pc = inputs[:,:,:3]
    if C > 3:
        nor = inputs[:,:,3:]

    centroid = np.mean(pc, axis=1, keepdims=True)
    pc = inputs[:,:,:3] - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    if C > 3:
        return np.concatenate([pc,nor],axis=-1)
    else:
        return pc
    
def plot_pcd_progress(filename, pcds, titles=None, suptitle='', sizes=None, cmap='Greys', zdir='y',
                         xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.4, 0.4)):
    rows = len(pcds)
    cols = pcds[0].shape[0]
    totals = rows * cols
    if sizes is None:
        sizes = [100 for i in range(totals)]

    if titles is None:
        titles = []
        for row in range(rows):
            for j in range(cols):
                if row == 0:
                    titles.append('center')
                else:
                    titles.append('sorted_center')

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(rows):
        # elev = 30
        # azim = -45
        pcd_row = pcds[i]
        for j, pcd in enumerate(pcd_row):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')

            # Scatter plot
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='blue', marker='o', label='Points')

            # Arrows
            if i==1:
                for k in range(len(pcd)-1):
                    x, y, z = pcd[k]
                    dx, dy, dz = pcd[k+1] - pcd[k]
                    length = (dx**2+dy**2+dz**2)**0.5
                    ax.quiver(x, y, z, dx, dy, dz, length=length, normalize=True, color='red', arrow_length_ratio=0.3)

            ax.set_title(titles[i*cols + j])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_pcd_uncertainty(filename, pcds, titles=None, suptitle='', sizes=None, cmap='Greys', zdir='y',
                         xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.4, 0.4)):
    total = len(pcds)
    rows = math.floor(math.sqrt(total))
    while(total % rows != 0):
        total +=1
    cols = total//rows

    if sizes is None:
        sizes = [0.2 for i in range(len(pcds))]

    if titles is None:
        titles = np.arange(len(pcds))
        
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(rows):
        elev = 30
        azim = -45
        for j, pcd in enumerate(pcds[i*cols:(i+1)*cols]):
            color = np.zeros(pcd.shape[0])
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=sizes[i*cols + j], cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[i*cols + j])
            #ax.text(0, 0, titles[i][j], color="green")
            ax.set_axis_off()
            #ax.set_xlabel(titles[i][j])

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    #plt.xticks(np.arange(len(pcds)), titles[:len(pcds)])

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
