import os, pickle
from zipfile import ZipFile
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import diefComp1Utils as util

# happening in SLS scaler
# MEAN = 1.82207407
# STD = 3.08905232
VALI = 0.2

RESAMPLE_U = np.array([-0.1702771, 31.32011473, -0.07542826, 0.04457935])
RESAMPLE_S = np.array([0.58785186, 41.41727523, 0.72031623, 0.16079391])


class datasetDIEF(Dataset):
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        """
        IMPORTANT!!! LABEL / TARGET / Y / TRUES IS NOT BINARY!!!
        -1 means negative
        0 means no label. In the metric calculation, mask the 0 entries.
        1 means positive
        flag: 'TRAIN', 'VALI', 'TRAIN_ALL', 'TEST'
        """
        self.args = args
        self.root_path = root_path
        self.file_list = file_list
        self.limit_size = limit_size
        self.flag = flag
        self.file_names = []

        # path of x and
        if flag == "TEST":
            partition = "test"
        else:
            partition = "train"
        self.pathX = root_path + partition + "_X.zip"
        self.pathY = root_path + partition + "_Y.csv"
        # load Y
        self.getitem = self.get_item_without_Y
        self.have_Y = False
        if os.path.exists(self.pathY) and os.path.isfile(self.pathY):
            print("datasetBBB:", flag, ": Has Y label.")
            self.have_Y = True
            self.labels_df = pd.read_csv(self.pathY, index_col=0)
            if flag == "TRAIN":
                self.labels_df = self.labels_df.iloc[
                    : -int(self.labels_df.shape[0] * VALI), :
                ]
            elif flag == "VALI":
                self.labels_df = self.labels_df.iloc[
                    -int(self.labels_df.shape[0] * VALI) :, :
                ]
            if args.test_run:
                self.labels_df = self.labels_df.iloc[: 3 * args.batch_size, :]
            self.aY = (self.labels_df.iloc[:, 3:].values).astype(np.float32)
            self.getitem = self.get_item_with_Y
        else:
            self.have_Y = False
            print("datasetBBB:", flag, ": No Y label. Y label is a zero array")
            self.aY = np.zeros(94).astype(np.float32)
        # load X
        zipX = ZipFile(self.pathX, "r")
        lFiles = zipX.namelist()[1:]
        self.lX = []
        if flag == "TRAIN":
            lFiles = lFiles[: -int(len(lFiles) * VALI)]
        elif flag == "VALI":
            lFiles = lFiles[-int(len(lFiles) * VALI) :]
        if args.test_run:
            lFiles = lFiles[: 3 * args.batch_size]

        for ifilename in tqdm(lFiles, desc="datasetBBB:init:load_X:" + flag):
            if ifilename[-4:] == ".pkl":
                d = pickle.loads(zipX.read(ifilename))
                # scale the time
                d["t"] = d["t"] / np.timedelta64(8, "W")
                if len(d["t"]) > args.seq_len:
                    i = np.random.choice(
                        np.arange(len(d["t"])), size=args.seq_len, replace=False
                    )
                    i.sort()
                    d["t"] = d["t"][i]
                    d["v"] = d["v"][i]
                # symmetric logarithmic standard scaler
                d["v"] = util.slsScaler(d["v"])
                ix = np.stack((d["t"], d["v"]), axis=1)
                # if test, we need to keep the base file name
                if flag == "TEST":
                    self.file_names.append(os.path.basename(ifilename))

                self.lX.append(ix)

        if self.have_Y:
            assert len(self.lX) == self.aY.shape[0]
            assert len(self.lX) == self.labels_df.shape[0]

    def get_item_with_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY[ind, :])

    def get_item_without_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY)

    def __getitem__(self, ind):
        return self.getitem(ind)

    def __len__(self):
        return len(self.lX)


class datasetDIEF_rs4H(Dataset):
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        """
        IMPORTANT!!! LABEL / TARGET / Y / TRUES IS NOT BINARY!!!
        -1 means negative
        0 means no label. In the metric calculation, mask the 0 entries.
        1 means positive
        flag: 'TRAIN', 'VALI', 'TRAIN_ALL', 'TEST'
        """
        self.args = args
        self.root_path = root_path
        self.file_list = file_list
        self.limit_size = limit_size
        self.flag = flag
        # path of x and
        if flag == "TEST":
            partition = "test"
        else:
            partition = "train"
        self.pathX = root_path + partition + "_X_v0.1.0.zip"
        self.pathY = root_path + partition + "_y_v0.1.0.csv"
        # LOAD Y
        # init with no Y
        self.getitem = self.get_item_without_Y
        self.have_Y = False
        self.file_names = None
        # self.column_names = None
        # check for Y
        if os.path.exists(self.pathY) and os.path.isfile(self.pathY):
            print("datasetDIEF_rs4H:", flag, ": Has Y label.")
            self.have_Y = True
            self.labels_df = pd.read_csv(
                self.pathY,
            )  # index_col=0)
            print("datasetDIEF_rs4H: original dfY.shape:", self.labels_df.shape)
            if flag == "TRAIN":
                self.labels_df = self.labels_df.iloc[
                    : -int(self.labels_df.shape[0] * VALI), :
                ]
            elif flag == "VALI":
                self.labels_df = self.labels_df.iloc[
                    -int(self.labels_df.shape[0] * VALI) :, :
                ]
            if args.test_run:
                self.labels_df = self.labels_df.iloc[: 3 * args.batch_size, :]
            print(
                "datasetDIEF_rs4H: dfY.shape after sub-partition:", self.labels_df.shape
            )
            self.aY = (self.labels_df.iloc[:, 1:].values).astype(np.float32)
            print("datasetDIEF_rs4H: aY.shape:", self.aY.shape)
            self.getitem = self.get_item_with_Y
        else:
            self.have_Y = False
            print("datasetDIEF_rs4H:", flag, ": No Y label. Y label is a zero array")
            self.aY = np.zeros(94).astype(np.float32)
        # load X
        zipX = ZipFile(self.pathX, "r")
        lFiles = zipX.namelist()[1:]  # do not use this one, because it is not sorted.
        print("datasetDIEF_rs4H: n files in zip:", len(lFiles))
        # TODO: Edit for competition when the file list is not provided
        if flag != "TEST":
            lFiles = self.labels_df["filename"].values
            print("datasetDIEF_rs4H: n files in dfY:", len(lFiles))
        else:
            lFiles = [os.path.basename(i) for i in lFiles]

            self.file_names = lFiles
        # We don't need the next block anymore,
        # since we already got the sub partition from dfY
        # if flag == 'TRAIN':
        #     lFiles = lFiles[:-int(len(lFiles)*VALI)]
        # elif flag == 'VALI':
        #     lFiles = lFiles[-int(len(lFiles)*VALI):]
        # if args.test_run:
        #     lFiles = lFiles[:3*args.batch_size]

        self.lX = []

        zipXnamelist = zipX.namelist()  # this function is VERY SLOW!
        count = 0
        for ifilename in tqdm(lFiles, desc="datasetDIEF_rs4H:init:load_X:" + flag):
            # if ifilename[-4:] != '.pkl':
            #     continue
            count += 1
            # if count > 1000:
            #     break
            if not partition + "_X/" + ifilename in zipXnamelist:
                print("datasetDIEF_rs4H: file not found:", ifilename, zipXnamelist[:3])
                ix = np.zeros((1, 4))
                self.lX.append(ix)

                continue
            d = pickle.loads(zipX.read(partition + "_X/" + ifilename))
            d["t"] = np.datetime64(0, "ns") + d["t"]  # convert timedelta to datetime
            d["v"] = util.slsScaler(d["v"])
            # RESAMPLE to 4H
            dtis = pd.DatetimeIndex(
                pd.date_range(start=d["t"][0], end=d["t"][-1], freq="4H")
            )
            iSeries = pd.Series(
                d["v"], index=pd.DatetimeIndex(d["t"]).tz_localize(None)
            )
            iSeries = iSeries.resample("4H")
            ix = np.zeros((len(iSeries), 4))
            ix[:, 0] = iSeries.mean().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 1] = iSeries.count().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 2] = iSeries.max().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 3] = iSeries.std().interpolate().reindex(dtis, fill_value=0).values
            ix = np.nan_to_num(ix)
            ix = (ix - RESAMPLE_U) / RESAMPLE_S
            self.lX.append(ix)

        # final checks
        if self.have_Y:
            if len(self.lX) != self.aY.shape[0]:
                print("datasetDIEF_rs4H: NOT EQUAL aY", len(self.lX), self.aY.shape[0])
                assert len(self.lX) == self.aY.shape[0]
            if len(self.lX) != self.labels_df.shape[0]:
                print(
                    "datasetDIEF_rs4H: NOT EQUAL df",
                    len(self.lX),
                    self.labels_df.shape[0],
                )
                assert len(self.lX) == self.labels_df.shape[0]

    def get_item_with_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY[ind, :])

    def get_item_without_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY)

    def __getitem__(self, ind):
        return self.getitem(ind)

    def print_shape(self, is_actually_print=True):
        str_return = "datasetDIEF_rs4H: x shape: " + str(self.lX[0].shape)
        if self.have_Y:
            str_return += ", y shape: " + str(self.aY[0].shape)
        else:
            str_return += "no Y"
        if is_actually_print:
            print(str_return)
        return str_return

    def __len__(self):
        return len(self.lX)


class datasetDIEF_rs10M(Dataset):
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        """
        IMPORTANT!!! LABEL / TARGET / Y / TRUES IS NOT BINARY!!!
        -1 means negative
        0 means no label. In the metric calculation, mask the 0 entries.
        1 means positive
        flag: 'TRAIN', 'VALI', 'TRAIN_ALL', 'TEST'
        """
        self.args = args
        self.root_path = root_path
        self.file_list = file_list
        self.limit_size = limit_size
        self.flag = flag
        # path of x and
        if flag == "TEST":
            partition = "test"
        else:
            partition = "train"
        self.pathX = root_path + partition + "_X_v0.1.0.zip"
        self.pathY = root_path + partition + "_y_v0.1.0.csv"
        # LOAD Y
        # init with no Y
        self.getitem = self.get_item_without_Y
        self.have_Y = False
        self.file_names = None
        # self.column_names = None
        # check for Y
        if os.path.exists(self.pathY) and os.path.isfile(self.pathY):
            print("datasetDIEF_rs4H:", flag, ": Has Y label.")
            self.have_Y = True
            self.labels_df = pd.read_csv(
                self.pathY,
            )  # index_col=0)
            print("datasetDIEF_rs4H: original dfY.shape:", self.labels_df.shape)
            if flag == "TRAIN":
                self.labels_df = self.labels_df.iloc[
                    : -int(self.labels_df.shape[0] * VALI), :
                ]
            elif flag == "VALI":
                self.labels_df = self.labels_df.iloc[
                    -int(self.labels_df.shape[0] * VALI) :, :
                ]
            if args.test_run:
                self.labels_df = self.labels_df.iloc[: 3 * args.batch_size, :]
            print(
                "datasetDIEF_rs4H: dfY.shape after sub-partition:", self.labels_df.shape
            )
            self.aY = (self.labels_df.iloc[:, 1:].values).astype(np.float32)
            print("datasetDIEF_rs4H: aY.shape:", self.aY.shape)
            self.getitem = self.get_item_with_Y
        else:
            self.have_Y = False
            print("datasetDIEF_rs4H:", flag, ": No Y label. Y label is a zero array")
            self.aY = np.zeros(94).astype(np.float32)
        # load X
        zipX = ZipFile(self.pathX, "r")
        lFiles = zipX.namelist()[1:]  # do not use this one, because it is not sorted.
        print("datasetDIEF_rs4H: n files in zip:", len(lFiles))
        # TODO: Edit for competition when the file list is not provided
        if flag != "TEST":
            lFiles = self.labels_df["filename"].values
            print("datasetDIEF_rs4H: n files in dfY:", len(lFiles))
        else:
            lFiles = [os.path.basename(i) for i in lFiles]

            self.file_names = lFiles
        # We don't need the next block anymore,
        # since we already got the sub partition from dfY
        # if flag == 'TRAIN':
        #     lFiles = lFiles[:-int(len(lFiles)*VALI)]
        # elif flag == 'VALI':
        #     lFiles = lFiles[-int(len(lFiles)*VALI):]
        # if args.test_run:
        #     lFiles = lFiles[:3*args.batch_size]

        self.lX = []

        zipXnamelist = zipX.namelist()  # this function is VERY SLOW!
        count = 0
        for ifilename in tqdm(lFiles, desc="datasetDIEF_rs4H:init:load_X:" + flag):
            # if ifilename[-4:] != '.pkl':
            #     continue
            count += 1
            # if count > 1000:
            #     break
            if not partition + "_X/" + ifilename in zipXnamelist:
                print("datasetDIEF_rs4H: file not found:", ifilename, zipXnamelist[:3])
                ix = np.zeros((1, 4))
                self.lX.append(ix)

                continue
            d = pickle.loads(zipX.read(partition + "_X/" + ifilename))
            d["t"] = np.datetime64(0, "ns") + d["t"]  # convert timedelta to datetime
            d["v"] = util.slsScaler(d["v"])
            # RESAMPLE to 10 minutes
            dtis = pd.DatetimeIndex(
                pd.date_range(start=d["t"][0], end=d["t"][-1], freq="10T")
            )
            iSeries = pd.Series(
                d["v"], index=pd.DatetimeIndex(d["t"]).tz_localize(None)
            )
            iSeries = iSeries.resample("10T")
            ix = np.zeros((len(iSeries), 4))
            ix[:, 0] = iSeries.mean().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 1] = iSeries.count().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 2] = iSeries.max().interpolate().reindex(dtis, fill_value=0).values
            ix[:, 3] = iSeries.std().interpolate().reindex(dtis, fill_value=0).values
            ix = np.nan_to_num(ix)
            ix = (ix - RESAMPLE_U) / RESAMPLE_S
            self.lX.append(ix)

        # final checks
        if self.have_Y:
            if len(self.lX) != self.aY.shape[0]:
                print("datasetDIEF_rs4H: NOT EQUAL aY", len(self.lX), self.aY.shape[0])
                assert len(self.lX) == self.aY.shape[0]
            if len(self.lX) != self.labels_df.shape[0]:
                print(
                    "datasetDIEF_rs4H: NOT EQUAL df",
                    len(self.lX),
                    self.labels_df.shape[0],
                )
                assert len(self.lX) == self.labels_df.shape[0]

    def get_item_with_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY[ind, :])

    def get_item_without_Y(self, ind):
        return torch.from_numpy(self.lX[ind]), torch.from_numpy(self.aY)

    def __getitem__(self, ind):
        return self.getitem(ind)

    def print_shape(self, is_actually_print=True):
        str_return = "datasetDIEF_rs4H: x shape: " + str(self.lX[0].shape)
        if self.have_Y:
            str_return += ", y shape: " + str(self.aY[0].shape)
        else:
            str_return += "no Y"
        if is_actually_print:
            print(str_return)
        return str_return

    def __len__(self):
        return len(self.lX)
