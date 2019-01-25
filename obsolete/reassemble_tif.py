import rasterio, os
import numpy as np
import cloudpickle

equi7_eu_crs = '+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs'


def reassemble_tif(dump_folder='/home/tle/temp/sbdsc_dump_test/dumpfiles/',
                   param='SM',
                   ref_raster='/home/tle/temp/sbdsc_dump_test/D20180501_050912--_SIG0-----_S1BIWGRDH1VVD_022_A0102_EU010M_E051N016T1.tif',
                   out_path='/home/tle/temp/sbdsc_dump_test/out_rasters',
                   out_name='TEST_SM',
                   tif_size=10000, block_size=10, write_file_px=2500,
                   mean_img=False,
                   n_images='all'):
    px_per_side = int(tif_size / block_size)

    if mean_img:
        # read dump to list
        dictlist = {}
        for file in os.listdir(dump_folder):
            if file.endswith(".dump"):
                dump_file = os.path.join(dump_folder, file)
                try:
                    with open(dump_file, 'rb') as file_file:
                        res = cloudpickle.load(file_file)
                except Exception as e:
                    print(dump_file, e)
                    continue

                value = np.mean(res.result[6][param])

                dictlist[file.replace('.dump', '')] = {'value': value}

        dataset = rasterio.open(ref_raster)
        transform = dataset.transform
        # read band 1
        in_array = dataset.read(1)

        for i in range(px_per_side):
            for j in range(px_per_side):
                k = str(i) + '_' + str(j)
                if k in dictlist.keys():
                    vl = int(dictlist[k]['value'] * 200)
                else:
                    vl = 255
                # replace value
                in_array[block_size * i:block_size * (i + 1), block_size * j:block_size * (j + 1)] = vl

        out_raster = os.path.join(out_path, out_name + '.tif')

        # write another raster
        with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
                           width=in_array.shape[1], count=1, dtype=str(in_array.dtype), compress='lzw',
                           crs=equi7_eu_crs,
                           transform=transform, nodata=255) as dst:
            dst.write(in_array, 1)

    else:

        # count number of dump files
        no_dump = len([name for name in os.listdir(dump_folder) if name.endswith(".dump")])

        # read dump to list
        time_dict = {}
        # since the counter starts at 1 and not at 0
        write_file_px += 1
        no_dump += 1
        count = 1
        for file in os.listdir(dump_folder):
            if file.endswith(".dump"):
                count += 1
                dump_file = os.path.join(dump_folder, file)
                try:
                    with open(dump_file, 'rb') as file_file:
                        res = cloudpickle.load(file_file)
                except Exception as e:
                    print(dump_file, e)
                    continue

                time_list = res.index
                value_list = res.result[6][param] * 200  # encoding

                px_name = file.replace('.dump', '')  # 2_2 string

                # add new value to the raster (ts stack), ignore the existing values
                cr_dict = {}
                for index, time in enumerate(time_list):
                    if n_images != 'all' and isinstance(n_images, int):
                        if index >= n_images: break
                    elif n_images == 'all':
                        pass
                    else:
                        assert False, 'n_images must be either "all" or a integer-number'

                    value = int(value_list[index])
                    cr_dict[px_name] = value

                    if time not in time_dict.keys():
                        time_dict[time] = {}

                    time_dict[time][px_name] = value

                    if count % write_file_px == 0 or count == no_dump:
                        for n_time, time in enumerate(time_dict.keys()):
                            filename = time.strftime("%Y_%m_%d_") + param + '.tif'
                            out_raster = os.path.join(out_path, filename)

                            first_time_write = count == write_file_px or (
                                        (count == no_dump) and (write_file_px >= no_dump))

                            if first_time_write:
                                print('generating file', n_time, 'of', len(time_dict),'for date' ,  time.strftime("%Y_%m_%d"), end='\r')
                                dataset = rasterio.open(ref_raster)
                            else:
                                print('appending to file', n_time, 'of', len(time_dict),'for date' ,  time.strftime("%Y_%m_%d"), end='\r')
                                # open raster
                                dataset = rasterio.open(out_raster)

                            transform = dataset.transform
                            # read band 1
                            in_array = dataset.read(1)

                            if first_time_write:
                                in_array.fill(255)

                            for cr in time_dict[time].keys():
                                # replace pixel value
                                j, i = cr.split('_')
                                i = int(i)
                                j = int(j)

                                in_array[block_size * i:block_size * (i + 1),
                                block_size * j:block_size * (j + 1)] = int(time_dict[time][cr])

                            # write raster
                            with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
                                               width=in_array.shape[1], count=1, dtype=str(in_array.dtype),
                                               compress='lzw',
                                               crs=equi7_eu_crs,
                                               transform=transform, nodata=255) as dst:
                                dst.write(in_array, 1)

                        time_dict = {}
            print('reading file', count, 'of', no_dump, '( writing tiff after', write_file_px, ')', end="\r")
            if count==no_dump: break




if __name__ == '__main__':
    param = 'SM'
    n_images = 'all'
    print('generating', n_images, 'images for ', param)
    reassemble_tif(dump_folder=r'D:\USERS\rq\rt1_test\dump_with_ndvi_savgol',
                   param=param,
                   ref_raster=r"E:\RADAR\E051N016T1\sig0\D20160101_050131--_SIG0-----_S1AIWGRDH1VHD_124_A0105_EU010M_E051N016T1.tif",
                   out_path=r'R:\Projects_work\SBDSC\data\codes\Qgis\testdata\temp\test_ndvi_savgol\SM',
                   write_file_px=60000,
                   mean_img=False,
                   n_images = n_images)

    pass
