# Code provided by Tomojit Ghosh(tomojit.ghosh@colostate.edu) and Michael Kirby (Kirby@math.colostate.edu)
#
# Copyright (c) 2020 Tomojit Ghosh  and Michael Kirby

# Permission is granted, free of charge, to everyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are .
#
# The software is provided "as is", without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.

import numpy as np
import sys
from utilityScript import *
from SupervisedCentroidencodeVisualizerPyTorch import SCEVisualizer
from sklearn import metrics
import pandas as pd


def createModel(dataSetName, dict2, trData, trLabels, tstData, tstLabels, params):

    # load data
    if dataSetName == "MNIST":

        trDataViz, trLabelsViz = trData, trLabels
        tstDataViz, tstLabelsViz = tstData, tstLabels
        annotDataTr = makeAnnotationMNIST(trLabelsViz)
        annotDataTst = makeAnnotationMNIST(tstLabelsViz)

    else:  # for USPS
        orgData, orgLabels = trData, trLabels
        nTrData = 8000
        trData, trLabels, tstData, tstLabels = splitData_n(orgData, orgLabels, nTrData)
        annotDataTr = makeAnnotationUSPS(trLabels)
        annotDataTst = makeAnnotationUSPS(tstLabels)

    num_epochs_pre = params["num_epochs_pre"]
    num_epochs_post = params["num_epochs_post"]
    miniBatch_size = params["miniBatch_size"]
    learning_rate = params["learning_rate"]

    standardizeFlag = True
    preTrFlag = True

    # build a model
    model = SCEVisualizer(dict2)
    print("Training centroid-encoder to build a model.")

    # train the model
    model.fit(
        trData,
        trLabels,
        learningRate=learning_rate,
        miniBatchSize=miniBatch_size,
        numEpochsPreTrn=num_epochs_pre,
        numEpochsPostTrn=num_epochs_post,
        standardizeFlag=standardizeFlag,
        preTraining=preTrFlag,
        verbose=True,
    )

    # reduce dimension of training and test data
    if dataSetName == "MNIST":
        pDataTr = model.predict(trDataViz)[len(dict2["hL"])].to("cpu").numpy()
        trCentroids = calcCentroid(pDataTr, trLabelsViz)
        pDataTst = model.predict(tstDataViz)[len(dict2["hL"])].to("cpu").numpy()
    else:
        pDataTr = model.predict(trData)[len(dict2["hL"])].to("cpu").numpy()
        trCentroids = calcCentroid(pDataTr, trLabels)
        pDataTst = model.predict(tstData)[len(dict2["hL"])].to("cpu").numpy()

    # calculate the MSE loss test data
    def findNearestCentroid(pData, centroids):
        nData = np.shape(pData)[0]
        nCentroids = np.shape(centroids)[0]
        dist = np.zeros((nData, nCentroids))
        for i in range(nCentroids):
            dist[:, i] = np.sum((pData - centroids[i]) ** 2, axis=1)
        return np.argmin(dist, axis=1)

    pDataTstLabels = trCentroids[findNearestCentroid(pDataTst, trCentroids)]
    # if label equal to trCentroids, then label should take location of trCentroids like 0,1,2,3,4,5,6,7,8,9
    pDataTstLabels = np.array(
        [np.where(trCentroids == x)[0][0] for x in pDataTstLabels]
    )
    MSE = metrics.mean_squared_error(tstLabels, pDataTstLabels)
    print("MSE loss on test data:", MSE)
    return MSE

    # now visualize the training and test data using voronoi cells
    # display2DDataTrTst(pDataTr,trCentroids,annotDataTr,pDataTst,annotDataTst,dataSetName)


def Model_Selecter():
    save_csv = "MSE2.csv"
    datasets = ["USPS"]
    error_funcs = ["MSE"]  # ["L1"] # ["HUBER"] #["HINGE"]  # Errors BCE
    num_epochs_pres = [25, 50, 100]
    num_epochs_posts = [50, 100, 200]
    miniBatch_sizes = [64, 128, 256, 512]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    nSamplePerClass = 1000
    tr_data_MNIST, tr_labels_MNIST, tst_data_MNIST, tst_labels_MNIST = (
        getApplicationData("MNIST", nSamplePerClass)
    )
    tr_data_USPS, tr_labels_USPS, tst_data_USPS, tst_labels_USPS = getApplicationData(
        "USPS"
    )
    # hyper-parameters for Adam optimizer
    MSE_df = pd.DataFrame(
        columns=[
            "DatasetName",
            "MSE",
            "hL",
            "hActFunc",
            "oActFunc",
            "errorFunc",
            "l2Penalty",
            "num_epochs_pre",
            "num_epochs_post",
            "miniBatch_size",
            "learning_rate",
        ]
    )
    for dataSetName in datasets:
        for errorFunc in error_funcs:
            for num_epochs_pre in num_epochs_pres:
                for num_epochs_post in num_epochs_posts:
                    for miniBatch_size in miniBatch_sizes:
                        for learning_rate in learning_rates:
                            if dataSetName == "MNIST":
                                trData, trLabels, tstData, tstLabels = (
                                    tr_data_MNIST,
                                    tr_labels_MNIST,
                                    tst_data_MNIST,
                                    tst_labels_MNIST,
                                )
                            else:
                                trData, trLabels, tstData, tstLabels = (
                                    tr_data_USPS,
                                    tr_labels_USPS,
                                    tst_data_USPS,
                                    tst_labels_USPS,
                                )
                            # parameters for the CE network
                            dict2 = {}
                            dict2["inputDim"] = np.shape(trData)[1]
                            dict2["hL"] = returnBottleneckArc(dataSetName)
                            dict2["hActFunc"] = returnActFunc(dataSetName)
                            dict2["oActFunc"] = "linear"
                            dict2["errorFunc"] = errorFunc
                            dict2["l2Penalty"] = 0.00002
                            # hyper-parameters for Adam optimizer
                            params = {}
                            if dataSetName == "MNIST":
                                params["num_epochs_pre"] = num_epochs_pre
                                params["num_epochs_post"] = num_epochs_post
                                params["miniBatch_size"] = miniBatch_size
                                params["learning_rate"] = learning_rate
                            else:  # for USPS
                                params["num_epochs_pre"] = num_epochs_pre
                                params["num_epochs_post"] = num_epochs_post
                                params["miniBatch_size"] = miniBatch_size
                                params["learning_rate"] = learning_rate
                            MSE = createModel(
                                dataSetName,
                                dict2,
                                trData,
                                trLabels,
                                tstData,
                                tstLabels,
                                params,
                            )
                            MSE_df = MSE_df.append(
                                {
                                    "DatasetName": dataSetName,
                                    "MSE": MSE,
                                    "hL": dict2["hL"],
                                    "hActFunc": dict2["hActFunc"],
                                    "oActFunc": dict2["oActFunc"],
                                    "errorFunc": dict2["errorFunc"],
                                    "l2Penalty": dict2["l2Penalty"],
                                    "num_epochs_pre": params["num_epochs_pre"],
                                    "num_epochs_post": params["num_epochs_post"],
                                    "miniBatch_size": params["miniBatch_size"],
                                    "learning_rate": params["learning_rate"],
                                },
                                ignore_index=True,
                            )
                            MSE_df.to_csv(save_csv, index=False)


if __name__ == "__main__":
    Model_Selecter()
