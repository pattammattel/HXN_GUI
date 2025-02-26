import tifffile as tf
import glob,os,tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path




def pyxrf_output_tiffs_to_image(wd, search_string ='detsum*quant*.tiff',
                             plot_norm_cbar = False,remove_edges = False, 
                             colormap= 'viridis',
                             scalebar= False,   
                             scalebar_params ={'color':'w','pixel_size':0.120,
                            'unit': 'um', 'font_size':15,
                            'length':6, "thickness":0.02,
                            'location':'lower left'},
                            label_xrf = True, norm_xrf_with_sum = False,
                            elem_line_str = (1,2)):
    
    xrf_tiff_files = [path for path in Path(wd).rglob(f'{search_string}')]
    #print(xrf_tiff_files[0])

    save_to_png = os.path.join(wd, "png_images")
    save_to_svg = os.path.join(wd, "svg_images")

    if not os.path.exists(save_to_png):
        os.makedirs(save_to_png)

    if not os.path.exists(save_to_svg):
        os.makedirs(save_to_svg)

    #color_schemes = [colormap for i in range(len(xrf_tiff_files))]

    for i, im_path in enumerate(tqdm.tqdm(xrf_tiff_files)):
        
        image = tf.imread(im_path)
        vsize, hsize = np.shape(image)

        if norm_xrf_with_sum:
            sum_tiff = tf.imread(os.path.join(image_path.parent, "SUM_maps_XRF_total_cnt.tiff"))

        if remove_edges:
            image = image[2:-2,2:-2]
            sum_tiff = sum_tiff[2:-2,2:-2]

        # Plot the image
        fig, ax = plt.subplots(figsize =(7,7))


        ax.imshow(image, cmap=eval(f"cm.{colormap}"))
        # Create a ScalarMappable object for the colorbar
        mappable = cm.ScalarMappable(cmap=eval(f"cm.{colormap}"))
        cmap_name = colormap
        ax.axis('off')  # Remove axis marks

        if plot_norm_cbar:
            mappable.set_array(image/np.nanmax(image))
        elif norm_xrf_with_sum:
            mappable.set_array(image/sum_tiff)
        else:
            mappable.set_array(image)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add color bar
        cbar = plt.colorbar(mappable, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=8)  # Adjust color bar tick label size

        str_ = im_path.stem.split('_')

        try:
            elem_name = f"{str_[elem_line_str[0]]}-{str_[elem_line_str[1]]}"

        except IndexError:
            elem_name = f"{str_[-1]}"


        if scalebar:
            logfile = os.path.join(im_path.parent, "maps_log_tiff.txt")
            resolution = scalebar_params['pixel_size']
            scalebar_pixels = np.ceil(scalebar_params['length']/resolution)
            #scalebar_pixels = np.ceil(hsize*0.2)
            label = f"{int(np.around(scalebar_pixels*resolution, -2))} {scalebar_params['unit']}"
            
            scalebar = AnchoredSizeBar(ax.transData, 
                                        scalebar_pixels,
                                        label, 
                                        loc=scalebar_params['location'], 
                                        pad=0.2,
                                        color = scalebar_params['color'],
                                        label_top = True,
                                        frameon = False,
                                        size_vertical = np.ceil(vsize*scalebar_params['thickness']),
                                        fontproperties = fm.FontProperties(size=30)
                                        )
            ax.add_artist(scalebar)

        if label_xrf:

            ax.text(int(vsize*0.08), int(hsize*0.08), 
                    elem_name, color='w', fontsize=30, fontweight='bold')

        # Save the plot as PNG
        output_path_png = os.path.join(save_to_png,(os.path.splitext(im_path.stem)[0]+f"_{cmap_name}"))
        output_path_svg = os.path.join(save_to_svg,(os.path.splitext(im_path.stem)[0]+f"_{cmap_name}"))
        plt.savefig(output_path_png+".png", dpi=600, bbox_inches='tight')
        plt.savefig(output_path_svg+".svg", dpi=600, bbox_inches='tight')

        # Show the plot (optional)
        plt.close()
        
def batch_img_conversion(wd,search_string ='detsum*norm*.tiff'):
    xrf_dirs = glob.glob(os.path.abspath(wd)+"/output*")

    for fldr in xrf_dirs:
        print(fldr)
        pyxrf_output_tiffs_to_image(fldr,
                                    search_string = search_string)
        
