
import json


mll_tomo_scan = {   
                "angle_info":(angle_start, 
                               angle_end, 
                               angle_num),

                "fly2d_scan":(dets,
                            x_start,
                            x_end,
                            x_num,
                            y_start,
                            y_end,
                            y_num,
                            exposure),

                "align_x":( True,
                            start,
                            end,
                            num,
                            exposure,
                            elem,
                            center_with,
                            threshold),
                
                
                
                "align_y":( True,
                            start,
                            end,
                            num,
                            exposure,
                            elem,
                            center_with,
                            threshold
                            ),
                
                "stop_iter":False,
                "add_angles":(False,[91,92]),
                "remove_angles":(False,[-90,-91]),
                "stop_pdf":False   
            }


with open("/data/user_macros/mll_tomo_params.json","w") as fp:
        json.dump(mll_tomo_scan,fp, indent=6)

fp.close()