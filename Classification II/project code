{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1uWvibF2fPfBpisuRY66MExCfZgXnXMNi",
      "authorship_tag": "ABX9TyMe4YxwbO1bgUk7CVxpqWxl",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JessicaaaJe/Jesscia_Data_Mining_Project/blob/main/Classification_II_project.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Classification II Project\n",
        "Jessica Chen"
      ],
      "metadata": {
        "id": "mw9wDh33c0U2"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "79ciefmZI13L"
      },
      "outputs": [],
      "source": [
        "'''\n",
        "Import for Classification II Project\n",
        "'''\n",
        "\n",
        "import math\n",
        "import random\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib import colors\n",
        "from matplotlib.ticker import PercentFormatter\n",
        "\n",
        "import numpy as np\n",
        "from numpy import mean\n",
        "from numpy import std\n",
        "\n",
        "\n",
        "\n",
        "from sklearn.metrics import get_scorer_names, accuracy_score, f1_score\n",
        "from sklearn.datasets import make_classification\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.model_selection import RepeatedStratifiedKFold\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "import sklearn.utils.parallel\n",
        "from sklearn import neighbors\n",
        "from sklearn.metrics import balanced_accuracy_score\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import classification_report, accuracy_score\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.decomposition import PCA\n",
        "\n",
        "\n",
        "from imblearn.ensemble import BalancedRandomForestClassifier\n",
        "from imblearn.over_sampling import RandomOverSampler\n",
        "from imblearn.under_sampling import RandomUnderSampler\n",
        "\n",
        "from IPython.core.completer import keyword\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "\n",
        "# load in the data set to be classified.\n",
        "x = np.loadtxt(\"/content/drive/MyDrive/share_folders_csc373/Data/heart_2020_cleaned_comb_num.csv\",\\\n",
        "                   skiprows=1, delimiter=\",\", dtype=\"int\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Basic Metrics Performance Measurement Function\n",
        "\n"
      ],
      "metadata": {
        "id": "dqLBgHJDX35I"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def TP(cm):\n",
        "    # this gives back the number of columns.\n",
        "    nc = len(cm[1])\n",
        "    tp = [0]*nc\n",
        "\n",
        "    # assign tp with the true positive values.\n",
        "    for i in range(nc):\n",
        "        tp[i] = cm[i][i]\n",
        "    return tp"
      ],
      "metadata": {
        "id": "MJ2qIeLcJHDs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def FP(cm):\n",
        "  # subtract the instances that were predicted to belong to the class by the True positive, we then get the false mnegative.\n",
        "    return [sum(cm[:, col]) - cm[col, col] for col in range(cm.shape[1])]"
      ],
      "metadata": {
        "id": "a3QCb0ApLoWe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def TN(cm):\n",
        "    total = np.sum(cm)\n",
        "    tp = TP(cm)\n",
        "    fp = FP(cm)\n",
        "    col_sums = np.sum(cm, axis=0)\n",
        "    row_sums = np.sum(cm, axis=1)\n",
        "    # tp[i] + fp[i] + row_sums[i]) gives the total count of\n",
        "    # samples that either belong to class i or were predicted as class i.\n",
        "    # subtracting this value from the overall total, we get the true negative.\n",
        "    return [total - (tp[i] + fp[i] + row_sums[i]) for i in range(cm.shape[1])]\n"
      ],
      "metadata": {
        "id": "oaDrTftVMJ80"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def FN(cm):\n",
        "  # sum of elements in the row minus the diagonal element gives the false negatives.\n",
        "    return [sum(cm[row]) - cm[row, row] for row in range(cm.shape[0])]"
      ],
      "metadata": {
        "id": "jqyFSkxmVYdH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def accuracy(tp, tn, fp, fn):\n",
        "    # accuracy = (tp+tn) / (tp+tn+fp+fn)\n",
        "    return [(tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]) for i in range(len(tp))]"
      ],
      "metadata": {
        "id": "7A-M7pY0VbTE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def F1(tp, fp, fn):\n",
        "    # F1 = 2*(precision * recall) /(precision + recall)\n",
        "    # precision = tp(tp+fp); recall = tp/(tp+fp)\n",
        "    precision = [tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] != 0 else 0 for i in range(len(tp))]\n",
        "    recall = [tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] != 0 else 0 for i in range(len(tp))]\n",
        "    return [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] != 0 else 0 for i in range(len(tp))]\n"
      ],
      "metadata": {
        "id": "vmGK_5F82CAU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def TPR(tp, fn):\n",
        "    # true positive rate\n",
        "    # tp = tp / (tp+fn)\n",
        "    return [tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] != 0 else 0 for i in range(len(tp))]\n"
      ],
      "metadata": {
        "id": "cSacDpdW2tZT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Macro Measurement Functions\n",
        " Macro-average gives each class equal importance, so we calcualted statistics by independently calculating for each class and then taking the average over the classes."
      ],
      "metadata": {
        "id": "aunLpCAC6WVe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def macro_TPR(tp,cm):\n",
        "    # an argument of tp list is given.\n",
        "    # fn represents the false negative list.\n",
        "    fn = FN(cm)\n",
        "    tpr_list = TPR(tp, fn)\n",
        "    # each class is weighted equally.\n",
        "    return sum(tpr_list) / len(tpr_list)"
      ],
      "metadata": {
        "id": "cbaIPo0u6V_G"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def macro_F1(f1):\n",
        "    return sum(f1) / len(f1)"
      ],
      "metadata": {
        "id": "qII_MEZX7XFl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def macro_ACCURACY(acc):\n",
        "    return sum(acc) / len(acc)\n"
      ],
      "metadata": {
        "id": "TL4qHn_j7Zuj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def class_conf_mat(cl, tp, tn, fp, fn):\n",
        "  # cl represents a specific class\n",
        "    return [[tp[cl], fp[cl]], [fn[cl], tn[cl]]]"
      ],
      "metadata": {
        "id": "JcSr0v-270CF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Micro-average Functions\n",
        "Micro-average is calculated by counting the total true positives, false negatives, and false positives across all classes."
      ],
      "metadata": {
        "id": "tLZiCKvJ7_pm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def micro_conf_mat(tp, tn, fp, fn):\n",
        "    # creates a 2*2 matrix with predicted positive on the first row and predicted negative on the second row.\n",
        "    mcm = [[sum(tp), sum(fp)], [sum(fn), sum(tn)]]\n",
        "    return mcm\n"
      ],
      "metadata": {
        "id": "_H4MB2Lo7hP1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def micro_TPR(cm):\n",
        "    # Computes the sum of all true positives from the confusion matrix.\n",
        "    tp_sum = sum([cm[i][i] for i in range(len(cm))])\n",
        "    # Computes the sum of all false negatives.\n",
        "    fn_sum = sum([sum(cm[row]) - cm[row][row] for row in range(len(cm))])\n",
        "    return tp_sum / (tp_sum + fn_sum)\n"
      ],
      "metadata": {
        "id": "OU8hJDny7_af"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def micro_F1(cm):\n",
        "    tp_sum = sum([cm[i][i] for i in range(len(cm))])\n",
        "    fp_sum = sum([sum(row) - row[i] for i, row in enumerate(cm)])\n",
        "    fn_sum = sum([sum(col[i] for col in cm) - cm[i][i] for i in range(len(cm))])\n",
        "    precision = tp_sum / (tp_sum + fp_sum)\n",
        "    recall = tp_sum / (tp_sum + fn_sum)\n",
        "    return 2 * precision * recall / (precision + recall)"
      ],
      "metadata": {
        "id": "ex_gLb7J9IN-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def micro_ACCURACY(cm):\n",
        "    tp_sum = sum([cm[i][i] for i in range(len(cm))])\n",
        "    total = np.sum(cm)\n",
        "    return tp_sum / total"
      ],
      "metadata": {
        "id": "tCXWi9SL9KBL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## KNN Classification"
      ],
      "metadata": {
        "id": "2QBstupc_FyB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Begin classification using\", MODEL_TYPE)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n5gfvonWOa0Y",
        "outputId": "b34dde21-126c-423d-f366-ddd6b483868e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Begin classification using KNN\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# set up the parameters\n",
        "RESAMPLE = None\n",
        "MODEL_TYPE = \"KNN\"\n",
        "TEST_SET_SIZE = 0.2\n",
        "\n",
        "# KNN algorithm is time inefficient. I choose to reduce the dimensionality to fast the classficiation.\n",
        "# IT retain 95% variance when using PCA\n",
        "PCA_COMPONENTS = 0.95\n",
        "KNN_N_NEIGHBORS = 5"
      ],
      "metadata": {
        "id": "6BS28cO9gcr3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Extract x and y from the loaded dataset.\n",
        "X = x[:, :-1]\n",
        "y = x[:, -1]"
      ],
      "metadata": {
        "id": "RRZhmGsYOf_e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=42)\n"
      ],
      "metadata": {
        "id": "yeUltkC1Oh5u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Apply PCA for dimensionality reduction and efficient classification.\n",
        "pca = PCA(n_components=PCA_COMPONENTS)\n",
        "\n",
        "X_train_pca = pca.fit_transform(X_train)\n",
        "X_test_pca = pca.transform(X_test)\n",
        "\n",
        "# Build up KNN Model.\n",
        "# n_jobs = -1 put all available CPU in the computer to fast the classification Process.\n",
        "knn = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS,n_jobs=-1)\n",
        "knn.fit(X_train_pca, y_train)\n",
        "y_pred = knn.predict(X_test_pca)"
      ],
      "metadata": {
        "id": "q2iUilhNOly0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### KNN Multi-Class performance measurement\n",
        "\n",
        "##### The result gives out a confusion matrix for the dataset that has 4 response variable (HD_kD) classes: 0, 1, 2, 3.\n",
        "\n",
        "The main diagonal [44942, 5, 172, 0] represents the number of correct predictions for each class, from class 1 to class 4, respectively. The other numbers outside this diagonal represent misclassifications."
      ],
      "metadata": {
        "id": "yutrj9TQjqC4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "NUM_CLASSES = len(set(y_train))\n",
        "\n",
        "conf_mat = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]\n",
        "err_cnt = 0\n",
        "\n",
        "# Use y_test for true labels, so the row represents the actual class.\n",
        "for i, val in enumerate(y_test):\n",
        "  # Use y_pred for predicted labels, the column represent the predicted class.\n",
        "  conf_mat[val][y_pred[i]] += 1\n",
        "\n",
        "print(\"Confusion matrix:\")\n",
        "for i in conf_mat:\n",
        "  print(i)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cF5UdHzjoswh",
        "outputId": "f8254c75-1278-4036-9b0f-6ed8349a6580"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Confusion matrix:\n",
            "[44942, 40, 559, 11]\n",
            "[1217, 5, 48, 1]\n",
            "[3630, 16, 172, 6]\n",
            "[535, 5, 32, 0]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# initialize c_mat to be assigned with values of confusion matrix.\n",
        "c_mat = [[ 8,  5,  20],\n",
        "         [ 2, 10,  10],\n",
        "         [ 5,  5, 270]]\n",
        "\n",
        "\n",
        "c_mat = np.array(conf_mat)\n",
        "\n",
        "#Determine overall accuracy\n",
        "total_cnt = 0\n",
        "total_correct = 0\n",
        "for i in range(NUM_CLASSES):\n",
        "  for j in range(NUM_CLASSES):\n",
        "    total_cnt += c_mat[i][j]\n",
        "    if i==j:\n",
        "      total_correct += c_mat[i][j]\n",
        "overall_acc = total_correct/total_cnt\n",
        "print(\"total count:\", total_cnt)\n",
        "print(\"total correct:\", total_correct)\n",
        "print(\"overall accuracy %3.3f\" % overall_acc)\n",
        "\n",
        "# get the statistics for each class.\n",
        "tp = TP(c_mat)\n",
        "fp = FP(c_mat)\n",
        "tn = TN(c_mat)\n",
        "fn = FN(c_mat)\n",
        "acc = accuracy(tp, tn, fp, fn)\n",
        "acc = [round(x, 3) for x in acc]\n",
        "tpr = TPR(tp, fn)\n",
        "tpr = [round(x, 3) for x in tpr]\n",
        "f1 = F1(tp, fp, fn)\n",
        "f1 = [round(x, 3) for x in f1]\n",
        "\n",
        "# -------macro meausrements-----.\n",
        "# Macro accuracy computes the metric independently for each class and then takes the average.\n",
        "macro_tpr = macro_TPR(tpr, c_mat)\n",
        "macro_f1 = macro_F1(f1)\n",
        "macro_acc = macro_ACCURACY(acc)\n",
        "\n",
        "\n",
        "# -------micro measurements-----.\n",
        "# Micro accuracy aggregates the contributions of all the classes to compute the average metric.\n",
        "micro_c_mat = micro_conf_mat(tp, tn, fp, fn)\n",
        "\n",
        "for i in range(2):\n",
        "  for j in range(2):\n",
        "    micro_c_mat[i][j] = round(micro_c_mat[i][j], 3)\n",
        "\n",
        "micro_tpr = micro_TPR(micro_c_mat)\n",
        "micro_f1 = micro_F1(micro_c_mat)\n",
        "micro_acc = micro_ACCURACY(micro_c_mat)\n",
        "\n",
        "\n",
        "print(\"\\nClass metrics.\")\n",
        "print(\"TP:\", tp)\n",
        "print(\"FP:\", fp)\n",
        "print(\"TN:\", tn)\n",
        "print(\"FN:\", fn)\n",
        "print(\"Accuracy:\", acc)\n",
        "print(\"TPR:\", tpr)\n",
        "print(\"F1:\", f1)\n",
        "\n",
        "print(\"\\nMacro metrics.\")\n",
        "print(\"Macro TPR: %2.3f\" % macro_tpr)\n",
        "print(\"Macro F1: %2.3f\" % macro_f1)\n",
        "print(\"Macro ACC: %2.3f\" % macro_acc)\n",
        "\n",
        "print(\"\\nClass confusion matrices:\")\n",
        "for i in range(len(tp)):\n",
        "  class_c_mat = class_conf_mat(i, tp, tn, fp, fn)\n",
        "  print(\"Class \", i)\n",
        "  print(class_c_mat)\n",
        "\n",
        "\n",
        "print(\"\\nMicro confusion matrix:\")\n",
        "for row in micro_c_mat:\n",
        "  print(row)\n",
        "print(\"\\nMicro metrics.\")\n",
        "print(\"Micro TPR: %2.3f\" % micro_tpr)\n",
        "print(\"Micro F1: %2.3f\" % micro_f1)\n",
        "print(\"Micro ACC: %2.3f\" % micro_acc)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F8Wy23DytYFF",
        "outputId": "a6b6e6a6-3d4a-4087-bd28-41ce2f224f3f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "total count: 51219\n",
            "total correct: 45119\n",
            "overall accuracy 0.881\n",
            "\n",
            "Class metrics.\n",
            "TP: [44942, 5, 172, 0]\n",
            "FP: [5382, 61, 639, 18]\n",
            "TN: [-44657, 49882, 46584, 50629]\n",
            "FN: [610, 1266, 3652, 572]\n",
            "Accuracy: [0.045, 0.974, 0.916, 0.988]\n",
            "TPR: [0.987, 0.004, 0.045, 0.0]\n",
            "F1: [0.938, 0.007, 0.074, 0]\n",
            "\n",
            "Macro metrics.\n",
            "Macro TPR: 0.000\n",
            "Macro F1: 0.255\n",
            "Macro ACC: 0.731\n",
            "\n",
            "Class confusion matrices:\n",
            "Class  0\n",
            "[[44942, 5382], [610, -44657]]\n",
            "Class  1\n",
            "[[5, 61], [1266, 49882]]\n",
            "Class  2\n",
            "[[172, 639], [3652, 46584]]\n",
            "Class  3\n",
            "[[0, 18], [572, 50629]]\n",
            "\n",
            "Micro confusion matrix:\n",
            "[45119, 6100]\n",
            "[6100, 102438]\n",
            "\n",
            "Micro metrics.\n",
            "Micro TPR: 0.924\n",
            "Micro F1: 0.924\n",
            "Micro ACC: 0.924\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### As indicated by the code below. Class 1 represents a significant majority of the data 88.83734552% while the other classes are under-represented in comparison, there is a class imbalance in the dataset. Usually, macro statistics would provide better evaluation on this dataset.\n",
        "I will test on K values from 1 to 21 to determine how these accuracy change\n"
      ],
      "metadata": {
        "id": "r8wyXHkRsJkf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Extract unique classes and their counts\n",
        "(unique, counts) = np.unique(y, return_counts=True)\n",
        "print(\"Unique Classes:\", unique)\n",
        "print(\"Counts:\", counts)\n",
        "\n",
        "# 2. Calculate the proportion of each class\n",
        "total_samples = len(y)\n",
        "class_proportions = counts / total_samples * 100\n",
        "print(\"Class Proportions (%):\", class_proportions)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-pu_H6e7rft7",
        "outputId": "0618c359-25d6-4e55-9be2-6e2f4b5e7442"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Unique Classes: [0 1 2 3]\n",
            "Counts: [227508   6633  19184   2770]\n",
            "Class Proportions (%): [88.83734552  2.59005447  7.49097015  1.08162986]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### KNN - test on different hyperparameters\n",
        "\n",
        "From the graph, we could see that all accuracies experience a significant increase when K approaches to 2, but stay relatively stable after that.\n",
        "And overall, micro statistics have the best performance."
      ],
      "metadata": {
        "id": "NCxhM-yYtoer"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# testing K from 1 to 20\n",
        "k_values = range(1, 21)\n",
        "\n",
        "# Store accuracies for each K value\n",
        "\n",
        "macro_accuracies = []\n",
        "micro_accuracies = []\n",
        "overall_accuracies = []\n",
        "\n",
        "for k in k_values:\n",
        "    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)\n",
        "    knn.fit(X_train_pca, y_train)\n",
        "    y_pred = knn.predict(X_test_pca)\n",
        "\n",
        "\n",
        "    # ----* calcualte macro accuracy *---- #.\n",
        "    NUM_CLASSES = len(set(y_train))\n",
        "\n",
        "    conf_mat = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]\n",
        "    err_cnt = 0\n",
        "\n",
        "    # Use y_test for true labels, so the row represents the actual class.\n",
        "    for i, val in enumerate(y_test):\n",
        "    # Use y_pred for predicted labels, the column represent the predicted class.\n",
        "      conf_mat[val][y_pred[i]] += 1\n",
        "\n",
        "\n",
        "\n",
        "    c_mat = [[ 8,  5,  20],\n",
        "            [ 2, 10,  10],\n",
        "            [ 5,  5, 270]]\n",
        "\n",
        "\n",
        "    c_mat = np.array(conf_mat)\n",
        "\n",
        "    # get the statistics for each class.\n",
        "    tp = TP(c_mat)\n",
        "    fp = FP(c_mat)\n",
        "    tn = TN(c_mat)\n",
        "    fn = FN(c_mat)\n",
        "    acc = accuracy(tp, tn, fp, fn)\n",
        "    acc = [round(x, 3) for x in acc]\n",
        "\n",
        "    macro_acc = macro_ACCURACY(acc)\n",
        "    macro_accuracies.append(macro_acc)\n",
        "\n",
        "\n",
        "    # ----* calcualte micro accuracy *---- #.\n",
        "    micro_c_mat = micro_conf_mat(tp, tn, fp, fn)\n",
        "\n",
        "    for i in range(2):\n",
        "      for j in range(2):\n",
        "        micro_c_mat[i][j] = round(micro_c_mat[i][j], 3)\n",
        "\n",
        "    micro_acc = micro_ACCURACY(micro_c_mat)\n",
        "    micro_accuracies.append(micro_acc)\n",
        "\n",
        "\n",
        "     # ----* calcualte overall accuracy *---- #.\n",
        "    total_cnt = 0\n",
        "    total_correct = 0\n",
        "    for i in range(NUM_CLASSES):\n",
        "      for j in range(NUM_CLASSES):\n",
        "        total_cnt += c_mat[i][j]\n",
        "        if i==j:\n",
        "          total_correct += c_mat[i][j]\n",
        "\n",
        "    overall_acc = total_correct/total_cnt\n",
        "    overall_accuracies.append(overall_acc)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(k_values, overall_accuracies, marker='o', label='Overall Accuracy')\n",
        "plt.plot(k_values, micro_accuracies, marker='x', label='Micro Accuracy')\n",
        "plt.plot(k_values, macro_accuracies, marker='.', label='Macro Accuracy')\n",
        "\n",
        "plt.xlabel('Number of Neighbors (k)')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('KNN Accuracy with varying k values')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "HKLeY-fI3BqC",
        "outputId": "7f221e20-71a5-4fe8-a735-2730fb1d48c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "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rlu3TuzcubPo5uYmurm5iU2aNBFfeeUV8eLFi6IoiuK1a9fEcePGiQ0aNBBVKpXo7e0tdu/eXfzrr7/K/JxEVLkJomjlq1WJiIgs1KZNG3h7e2Pnzp32LqVS27hxIwYOHIh9+/Y98BT05khJSUHNmjXx4YcfGl0DRURUHfGaKyIisqu///4bp0+fxqhRo+xdSqW3aNEihIaGPvD086aYmkFTd72dudOqExFVdbzmioiI7OLcuXM4ceIE5s6di8DAQKPrkch8a9aswdmzZ/HHH3/gyy+/rJAZENeuXYtly5ahX79+qFGjBg4cOIDVq1ejd+/eeOSRR6z+fkRElRHDFRER2cVvv/2G999/H40bN8bq1atLnWSESjdixAjUqFED48ePx8svv1wh79GyZUsoFArMmTMHaWlp+kkuOK04EVEhXnNFRERERERkBbzmioiIiIiIyAoYroiIiIiIiKyA11yZoNVqcefOHbi7u1fIRcFERERERFQ5iKKI9PR01K5dGzJZ6X1TDFcm3LlzB8HBwfYug4iIiIiIHMTNmzdRp06dUrdhuDLB3d0dgNSAHh4edq6malOr1di+fTt69+4NpVJp73KqBba5bbG9bY9tbntsc9tie9se29z2HKnN09LSEBwcrM8IpWG4MkE3FNDDw4PhqoKp1Wq4urrCw8PD7v9wqgu2uW2xvW2PbW57bHPbYnvbHtvc9hyxzc25XIgTWhAREREREVkBwxUREREREZEVMFwRERERERFZAcMVERERERGRFTBcERERERERWQHDFRERERERkRUwXBEREREREVkBwxUREREREZEVMFwRERERERFZAcMVERERERGRFTBcERERERERWQHDFRERERERkRUwXBEREREREVkBwxUREREREZEVMFwRERERFbV7NrB3jul1e+dI622NNZnPEetiTeZxxJosxHBFRERUXTjiiYsj1iSTA7s/Mq5r7xxpuUzOmhy1JketizVV3pospLB3AURERA9s92zp/3S7TjZet3cOoNUA3aeyJt2JC2BYl+7Epfu7tq3HUWvS1VG0rqL1mPqd2rAmmUYDIByy/Z8D+z5xiJr0r+3dTo5alzVrEkVA1JbxEKX/xpS2TbMngcwkqYaMBKDtGODEUuD4j0C78UDD3sCdU9KxIAIiCn6KD/jTxHFErfTcLxxoPlj6nsdHwzc3DLL90fb9nluI4YqIyJE54gm6I9ZU9AS90xuG9TA0FIp8S/r97P4IyM8FHnkVODAfODAP6PyGdEKVmVR4cmZwAqQ1fTIkikC+Gm45d4Gky4BCXvJ2RscB0KAHkHpLqin1JtBqBHBqBXB6JdByOFD7IeDStoITRU2Rn9pir4ucSBptqzGxXFvKcQqW+zeX6tozW9retymQEAP8OragQUXT7SyWsLyk7S3Zx6cx5Ps+weMQIEAEfBoBd88Aq5823F5/vNJeW7KtqdcFPOtK7bT7Y2kbjzrAlb+Ay9sNg4D+949ir0XTr4t/Z0rdp/gxtYBQ8G9Q9+9QkAP750oPCIAgSD+BUp5L/6MA0FethuKCcxnbl/Hc2cOwJqcawLFFwNHvpe+dwecv4VERji+SHjp/L5YediSP/h0dAQhXUWmCFcBwRUTk2BzxBN3eNWnygfwcKRzkZ0s/G/cD0u5If+1MvAz/rCDINmwEzq+T/grq2wQ4v8HwRL/oX3iNTv6KneCVuK6EE0LdunqdpDa5uhOo0x6IPQrcOiYFhtRbwMZXpJN5rQbQ5ksPUVvwvJRlYtF9NMW2yS88plhsm6In6vs/lx46B76QHuWgBNATAGLK9yvVO/mz9NA5u0Z62JvuhDYxRno4AEH3u0y6JD0cQkFNabekh6MRNUC+ply7CgCcACA705oVAXkZ0qPCCIAgK+VRsD77XuEuNfxhED6NfqLYa1kp25a1r4mfggyAAPHmUQgQIcqdIFSSYAUwXBERFXLEHpmKHt4iljB0w6i3ociyDhMAdbZUgzoLiHgROPglcOQboN04IKQrcG2PFHrUBeEnP6fIo+jyIuvVOeZtp80v9SPJz/2Ch4suOLdOethT7BHpoXPnpPRwaKZPeIxPhmQQBSA/XwOFUgmhxJOtEk7Aij6/f6PwvX2bSP8eBUHqcZDJi/2UmVhecMJo1rYFxzbaVl54jOv7gKu7pOWiBgjrBYT1LNJEgul2M9mcJSw3e7uC5Ze3A5e2QgsZZNACjfoCjaKK7StY8NqSbU3VKQDRm4Do3wGZQvr32Www0PzJwu+N/rsjkw5n8Fow/bro962kbYxeAwah4ej3wOGvAbkS0KiBjhOBiBek7Yx67nT/vUORZdAvV+fnY9++PegS2QVKhbzkY5g8XpHnJ3+Wht7JlIBWLfUWtxtX7LtbrB3KCkZlbVMW3f+nyJ0ATR7Q/jn79xLtnQPh5hFoBAXkmjypRnvXZCaGKyKyD0cMMrbokdFqgNw0ICethJ+phT+LrnP1MRxKonQBjnwrBRrdkJjShnCVtOxBFe/t+HuJ9LAVmVJqC4UzoFABChXE5CvSXzshQKjTDoUnYEVPSIqfnJk6WRGKrS+2rdHJoKkTyYLH0W8LhylFTpJOQnUn8jJ5sdeKB1imW17GskNfSb1WupOprlOBrv+FwcmsBfLVamzZsgX9+vWDUqks3++y+Ale8yftezK1d44UrHR/xNDVF9zBfnXtnQNc2gpNlyn4X3o4HnOPhnzfJ0DQQ/atKfp343bya2L/39/hr43rUnmWry61Ghmqy4BvY+BBvuMnlhrX5B5g399f0T/W6V4Ddq/J4Htu75oswHBFRPZh76FlppTVS9R5EpB1r5RwlGq0XJ6diu7Jd6C48jaQm2694R/qbOnhSFy8pYCjVBUEHWdAUST4mFpeLBjp15e6vMjr4jNH7Z0DYfdH0l87xXzpgmx7/5/x3jlSsNKFBrmT/U86939ufDIlk9n9ZMphTvBM9Q6b+u+DnWrSdnoD2LIF2si3IJeX8N9SG9fkMO3kqHWxJotrcpjvuYUYrojI9kRRGnaQHif9x/L2CWko2ZW/pGtTQrtLf2Xf91nZF/aaO2uS/mJ2M45RM9Tw4mynGtJF/7r/sFtABsADAHKKrVCopAubVR6FP1WeBc89jddd3AKcXlU4lKTDBGk4idFQK5hYVsqwLoNlKHk7U8O69s0F9s4uDA0Pv2T/0OBof+1kaKicNWk1pofd6l5ry3fdjtVqUqsdr6ai7FmT7n0drS7WZHlNjvI9txDDFRFZX34ukHZbumBf/7gJpBZZpi5yUfClrdJD59pu6WF3BcPmivc2KVwMg4/BT8NglK9ww9EzMejQpSeUbt6F6xVO5pexd44UrIqfoLv52jc07J3tkKHBYf7aydBQeWsqbUiyvf7NsSbzOWJdrMk8jliThRiuiKzJEa8jsnZdWi2QlVQQlnTB6bbh68wE847l5gt41pGmERa1Us9Iy2GlXKhb5IJ1a1zsKyth+cU/gZhNhRdntxsHdPoP4OwphSa5+ePtRbUaSdcEILB1+cbpO+IJuiPW5Ih/7WRoMI8j1kREVE4MV0TW5IjXERWvq6x7AOVlFutxKtLzlHZbClKa3LLfU+kqBSePIOmnZ3DBz4KHR23p+pm9c6SbFOqGlnmH2n9oWcwmExccB9qnLkc8QXfEmhzxBN0RayIiogrFcEVkTfa4K3zx64UMrivSSOvbjgVyM6R7AGXeQ80MH8jWjQMubAKC2knh5rvOUojKvm/GmwpS2CgaljyDAc8iQcqlZtkzjvF6lLI54gm6I9ZERETkABiuiKyty3+BjATDCRHcawMX/gBiNpcdhIyWaaWheMWX6fa3cDpt+bFv0aXogtt/G2/k7FkkNJnoeXIPtGhonEmOGGQcsUeGiIiIKg2GKyJrEEWp9ydmk3QzxXtXdSukH+l3pIddSTfLFLX5EADpHkAthxYZplckTKk8K74cRwwy7JEhIiKiB8BwRVReWi1w65gUpmI2A6mxhesEudTDJJNLIaHFUKDFkGKTKBSZnEEmNzHZQtFlcmmIndGyotsKJRxTt04wvgdQrTAOLSMiIiKyEoYrMo+jzoJna5p84N+DUg9VzP+AjLjCdUpXoGEvAAIQvcH4OiKfhvafqMHR7gFEREREVIUwXJF5HHUWPFvIzwOu7wWiN0o3cs1KLlzn7AE06gOEPw40eBQ4/LXjXUcEOOY9gIiIiIiqGIYrMo89ZsGzJ3U2cGWn1EN1cSuQm1q4zsUbaNIfaPo4ENoVUDgXrnPE64h07+to9wAiIiIiqmIYrsh8RQOWLmTVaS/dCPbfw4BfE2n67coqNx24vF26huryDkCdWbiuhj/QdIAUqOo9AshL+KfjqNcROWpdRERERFUIwxVZpu3YwmAFALeOSw+dGv6AbxPp4dek8Lmrt+1rNUf2feDin1KgurrL8Oa4nsFSmAp/HKjTQZqIgoiIiIioBAxXZJk1Txc8kSbzRnAE4OwOJF4EUm8CGfHS4/pew/3c/AzDlm8TwK8poHS39ScAMhKBC/+Thvxd3wdo8wvXeTeQwlTTx4Habcq+CS4RERERUQGGKzLf9mnS1OMA8MxvwJ2ThddcjVwnDatLvAQkxgCJF4CEC9LP1JtAZgJwPUEKM0Uo3HzRSfCBbOteICC8IHg1BdxqlV2PJTMYpt2RpkuP3gTEHpJuwKvjF17YQ+UXzkBFREREROXCcEXm2TsHOPSV9Dz4YSDsUaBhT+l10Uku6rSVHkXpQ9eFguB1UQpeqbEQMhPhi0TgRIzhPq4+Us+Wb2PDni43n8JtyprB8OGXgYMLpB6qokMXASCwdUEP1ROAT9gDNQ0REREREcBwRebKTJJuSCtqgUffK+zdMWe2OWf3EkJXBvLjzuPszl/RqrYz5MmXpfCVEgtkJQE39kuPolxrST1buiGGrZ8xDFhbJgPHvpeu/TryjeG+wRFSD1XTAUDNeuVvCyIiIiIiExiuyDy5aVKwCu0O1O9suK68s80514BY+yHcrBWHFo/2g1yplJbnZUq9W4kXi/R0xQAp/0r3mPr3gPQoqugMhoB03Zcgk2b2C38CaPIY4BFYvjqJiIiIiMzAcEVlS7wInF0rPe/xXsW/n5MbEPSQ9CgqLxNIulQYthILrum6/y8AsXC7sJ5SD1WT/obDCImIiIiIKhDDFZVt90dSr1Xj/sZD+2zJyU2awa92G8Pluz4E9n0GyBTSzH/BEUDb0fapkYiIiIiqLd64h0p39wwQvRGAAPR4197VGNs7RwpW3d8FpidLP3d/JC0nIiIiIrIh9lxR6XZ9KP1sPhjwb2bfWorTzQrY/d3C6750P03NIkhEREREVIEYrqhksUeBy9sBQQ50f8fe1RjTagyDlY45MxgSEREREVkZwxWZJorArg+k562fBmo1sG89puhuEGwKe6yIiIiIyMZ4zRWZdm2PdI8puRPQ9W17V0NERERE5PAYrshY0V6rtmMBr2D71kNEREREVAkwXJGxi38Ct08AChcg8k17V0NEREREVCkwXJEhrbZwpr2IFwB3f/vWQ0RERERUSTBckaHz64H4c4CzB/DIa/auhoiIiIio0mC4okKafGDPbOl5x4mAq7d96yEiIiIiqkQYrqjQ2TVA8hXAxRt4+CV7V0NEREREVKkwXJEkPxfY86n0vPMbgMrDvvUQEREREVUyDFckOfkzkBoL1AgAOjxv72qIiIiIiCodhisC8rKAfZ9Jz7u8BShd7FsPEREREVElxHBFwPFFQEY84FUXeGi0vashIiIiIqqUGK6qu5w04MAX0vOuUwCFk33rISIiIiKqpBiuqrsj3wDZ94FaDYGWw+xdDRERERFRpcVwVZ1l3QMOfS097z4VkCvsWw8RERERUSXGcFWdHZwP5KUD/i2A8EH2roaIiIiIqFJjuKqu0uOAoz9Iz3u8C8j4VSAiIiIiehA8o66u9s8F8rOBoHZAoz72roaIiIiIqNJjuKqOUmKBv5dKzx99DxAE+9ZDRERERFQFMFxVR3s/BbRqoH4kENrN3tUQEREREVUJDFfVTdIV4PRq6fmj0+1bCxERERFRFcJwVd3smQ2IGqBhFBDcwd7VEBERERFVGQxX1Un8eeDcOul5j2n2rYWIiIiIqIphuKpOdn0EQATCBwKBLe1dDRERERFRlcJwVV3cOgFc/AMQZED3d+1dDRERERFRlcNwVV3s+kD62XI44NvIvrUQEREREVVBDFfVwY0DwLXdgEwJdHvb3tUQEREREVVJdg9XCxcuRP369aFSqRAREYFjx46VuK1arcb777+PBg0aQKVSoVWrVti6desDHbPKE0VgZ0Gv1UOjgJr17VoOEREREVFVZddwtXbtWkyaNAkzZszAyZMn0apVK0RFRSEhIcHk9tOmTcP333+Pr776CtHR0XjxxRcxaNAgnDp1qtzHrPKu/AXcPAIoVECX/9q7GiIiIiKiKsuu4WrevHl4/vnnMXbsWISHh+O7776Dq6srlixZYnL75cuX45133kG/fv0QGhqKl156Cf369cPcuXPLfcwqTRQLr7Vq/xzgEWjfeoiIiIiIqjCFvd44Ly8PJ06cwNSpU/XLZDIZevbsicOHD5vcJzc3FyqVymCZi4sLDhw4UO5j6o6bm5urf52WlgZAGoaoVqst/3AOQriwGYq7ZyA6uSH/4f8ADvhZdO1bmdu5smGb2xbb2/bY5rbHNrcttrftsc1tz5Ha3JIa7BaukpKSoNFo4O/vb7Dc398fFy5cMLlPVFQU5s2bhy5duqBBgwbYuXMn1q9fD41GU+5jAsDs2bMxa9Yso+Xbt2+Hq6urpR/NMYha9LjwLtwBXPLuiQt7jtq7olLt2LHD3iVUO2xz22J72x7b3PbY5rbF9rY9trntOUKbZ2Vlmb2t3cJVeXz55Zd4/vnn0aRJEwiCgAYNGmDs2LEPPORv6tSpmDRpkv51WloagoOD0bt3b3h4eDxo2XYh/PMLFKfvQFR5IfSZLxCqcszPoVarsWPHDvTq1QtKpdLe5VQLbHPbYnvbHtvc9tjmtsX2tj22ue05UpvrRrWZw27hysfHB3K5HPHx8QbL4+PjERAQYHIfX19fbNiwATk5OUhOTkbt2rUxZcoUhIaGlvuYAODs7AxnZ2ej5Uql0u6/zHLRqIH9cwAAwiOvQeley84Fla3StnUlxja3Lba37bHNbY9tbltsb9tjm9ueI7S5Je9vtwktnJyc0LZtW+zcuVO/TKvVYufOnejYsWOp+6pUKgQFBSE/Px/r1q3DE0888cDHrFJOrQDu3wDc/ICIF+xdDRERERFRtWDXYYGTJk3C6NGj0a5dO3To0AHz589HZmYmxo4dCwAYNWoUgoKCMHv2bADA0aNHcfv2bbRu3Rq3b9/GzJkzodVqMXnyZLOPWeWpc4B9n0nPI98EnNzsWw8RERERUTVh13A1bNgwJCYmYvr06YiLi0Pr1q2xdetW/YQUsbGxkMkKO9dycnIwbdo0XLt2DTVq1EC/fv2wfPlyeHl5mX3MKu/vJUDabcAjCGg7xt7VEBERERFVG3af0GLixImYOHGiyXV79uwxeN21a1dER0c/0DGrtNwMYH/BPb+6TgaUqtK3JyIiIiIiq7HrTYTJyo5+B2QlATVDgNbP2LsaIiIiIqJqheGqqshOAQ4tkJ53fweQcyYbIiIiIiJbYriqKg59BeSkAr5NgeaD7V0NEREREVG1w3BVFWQkAke+lZ73eBeQye1bDxERERFRNcRwVRUc+AJQZwK12wBNHrN3NURERERE1RLDVWWXehs4/qP0vMc0QBDsWw8RERERUTXFcFXZ7fsM0OQCdTsBDR61dzVERERERNUWw1Vldu86cGq59PzR99hrRURERERkRwxXldmeTwBtvtRjVa+TvashIiIiIqrWGK4qq4QLwNm10vMe0+xbCxERERERMVxVWns+BiBKswMGPWTvaoiIiIiIqj2FvQugcrh7BojeCEBgrxURERGRCRqtiGPX7yEhPQd+7ip0CPGGXMbr04tzxHbSaEUcvX4PJ5IE1Lp+Dx3D/Oxek7kYriqjXR9KP1sMAfya2rcWIiKiKoonnebX5GjttPXcXczaHI27qTn6ZYGeKswYEI4+zQPtVpejtZUjtpNhTXL8fPlvu9dkCYaryib2KHB5OyDIgW5T7F0NERFVQo52gueINfGkszw1SRyhppdWnIRYbHlcag5eWnES3458yC61OVpbOWI7OWJNlmK4qkxEEdj1gfS8zUigVgP71kNE5GD4V/2yOdoJniPW5IgneKzJPBqtiFmbo41qAgARgABg1uZo9AoPsOm/Q0drq4pqJ1EUoRULf2pF6R20xZaLoghRLLIcIvI1IqZvPO9wvztLMVxVJtf2ADf2A3InoOtke1dDRORQ+Fd98+pxpBM8R6zJnJPOmZvO4+HQWpDLBKPtRKMFRZ+KpW4rGqwrfKXRln7SCQAzNp1HiyAvyGTSccUix9AdSiw4idW/LthG1G8jFtnOxPZFXmu0It7dcK7Umt79/RxqOCkgCpBOtLUitKIIjbbwxFv3WndMbcFJt6ZgnbZgW9063cm4tI8IjRZFlouIvZdt8O/NVG13U3Pw2ppTCPJyAQRAgABBkH63QtHXgqBfptVqceWWgOt7rkEhl0EouLeoTDC9L1C4vwgRX/51udS2+u9vZ3ExLl3fLhptkUeR17p1+VqpbTQioNFqC9YXPC9o63ytFlqt1Jb67Qse6blqs9qp5axtkAuC/juha2dRLPw+6Go2+u5bma6mY9fvoWODWhX7Zg+A4aqyKNpr1W484FnHvvUQkU05Wu+Ho9XkaCfojliTRiti5qaSQwMATN94Ho39PaSTyYKTWBQ5edJqC0+iip7oqtVqXE0Djl6/B7lcYfBXa8PtdftLz/M12jJDw+TfzuJKYgZELaDWisjXSCeSao108phfsCxfI0KtFaHRaqV1Gt26otuJUBfsn6+VnkvrC7bRiMjN10CtKfksUQQQl5aL1u/veJBfh9XFp+XikU932bsMA8mZeRi55Ji9yzDpf2fvlmMvObbcvGL1WgAgPScfX/x1uUKO/SAyczU2eR+ZgCIhv3QJ6SWHQkfAcFVZXPwTuH0CULoCkZPsXQ0R2ZCj9X44Wk3WGt6iG6ai+ytx0b+Qawv+eqwLGBqx8K/AhX99L/zrulqjxbu/l/5X/Snr/0FqthoaLaDWaAseov55nkYLdb500q/WaJGXL5rcTtpWChO6dXn50vN8rQh1vnSsvHytyXqKSkjPRfe5e8xpdhMUwPm/y7lvydJy8vH5tktWP25VJBMAuUyAIHXH6HtTgOK9M1KPSsGKwmUFzwHDXhsU7Y0pWJaj1iI1W11mTYEeKni6KiGXCZAJAmQyQapT0L2Wen/kMgGCULhOEATIC9bJCvaVCzA8RsE+8oL9ZDIB8ak52HIursy6HmsZiEBPVSm9dKK+p0Ys6B2K/TcWwXWDIQgyg+11w9pQpMew8JjArXtZOHUzpcyaOjaohQa+bpALAuQymfT5ZdLnU8gE/XO5vOCnrPAhK7aNQi7o21UuK9xeJpO2u3A3DR/8EVNmTZ8PaYWH6nrpfze675FMJn0/ZAXLofvdCIXLBRmKbKPrDYThNgU9hABw+GoyRiw6UmZNfu6qMrexJ4arykCrBXZ/JD2PeBGo4WffeojIZhyt98MWNYmiiNx8LTJy85GRky/9LHiemZeP9Jx8ZOYWLr+WmGHW8JY272+HXCYYDEkqOsxGW8FDWopLyVLj7XX/2PZNzeAkF+CkkOtPgmSC4UmQTDA8WdKdKGVnZaJGjRr6Ez39/jLd/roTqsL9kzNzcTUxs8yaIkK8EepbAwqZdNKolMsglwlQygQodM/lAhQyGRRFfirl0kmqbjvd/oXbFTlWwfKzt1Pw6urTZdb087j2iAiVhiYViSrS62IZXjBYJ5S4rvi+lp50rnzuYZsNlzK3pnnDWtt0CJdGK+LUp7sQl5pj8o8JAoAATxW+HN7Gop52tVqNLVtuoF+/ZlAqlRbVZG5bvdqjoc3a6uHQWvjxwPUy22lQmyCbjUjoEOKNQE9VmTV1CPG2ST3lxXDliHbPBmTywuuqzq8H4s8Bzp6F67tPtV99RGQTjnhhdlk1AdLQskBPF+SoNYXBKLcgEOXkIyNXg4xcNTJzNUg3WF64XX4FJJ20nPwHPoZQ7C/u8oK/ouv/Kl/w1/ZctRYpZvxVv2mgO+rUdIWyIDDoHk5yKQzonivlMigVUjhwUsiKbCsYPHcqsp1SLiuyrYCzt1Lx8sqTZdb007gIi0/wpBPPLejX7xGLTjzNPel8vWcjm510Bnu7YvaWC2We4D0S5lutTzodsSZA6s2aMSAcL604WXC9k2FNADBjQLhNhzA7Yls5Yjs5Yk3lwXDliGTywp6qzpOAPbOl57VbAQfmAd3ftV9tRFbmSNftOEpNeflaJGXkYteFBLN6ZJ758QhquTkbXdsiFrkmxtR1MobXxRRurx/mYmL/rLx8xKflllp/Qnounlh40CptUcNZATdnOWo4K6SHSgE3J+mnbllyZh7WHr9Z5rE+HdwCD9WtWWSIUUFAMghG0nKhIDgVriscgmQOc0PD9Mea2Sw0BHq6ONwJHk86WVNF6NM8EN+OfMho6HKAnYYuO2pbOVo7OWpNlmK4ckS6HqvdHwF3TgPJVwClC3B9nxSsOFMgVRGOdN1ORdckiiJSs9RISM9BYnouEjNykZAm/UxMlx66dfezyu7xKOrItXvlrquieKgU8HF31gcgN2cF3At+Fg1G+nWqgnVFQpSrUg6ZGScbGq2IfZcSyzxBf6ptcLXuaXDEEzxHrAlwzBM81mR5bb3CAxzmj3eO2laO1k5Fazp8JQHb9x9F78gIh7ithrkEsehcnwQASEtLg6enJ1JTU+Hh4WG/QnZ/DOz9tPB1FQxWhUNJ+lk8hpnKx1HavKTrdnT/6XSka4lKqyk3X4OkjDwkpJkOTQlpOYhNSEGGRlbqDGTFKWQCPFwUuJdZdtAa3bEeQn1rSBcbC4LRdTLFr3spvOYFRa6DkbYpcf+CZefvpGHGpvNl1rT6edtd+wEU/u4A0yfo9vw+OVJNurqs/QeEB/3viiP+oQWwfy92STU52kmnI7aTtVnr/zurQ1tZi6OcrwCWZQP2XDmyblOB/XMBbT7vbUVViqNcS6SfDlorzco2Y1PpU0K/+csZbPnnrhSmCnqbzJktC0X+Ju/pooSvuzN8azjDz0P66euue66S1rk7w8tFCRFAZzMuzJ4+oJnN/s+5Td2a+G7vVYfqkQEc86/CjliTri5H/Uu1I9UESD1rjnY/HblMQESIN5JjREQ4QBvpanK0dnJUbKuqj+HKke37rDBYafKAvXMYsKhKOHb9nlnXEg357jC8XJVGNz8sOlV2vqZwtrfiN1s0fcPFoussqzszT4NNZ4zvjeIkl8HX3Rk+JkJTLVcFLp39G09EdUeAlxtUSrlF7+loQ6YcdRgX4JhDSRgazOeINRERWYrhylHtnSNdc6UbCqh7DTBgUaWTkJ6D6DtpiL6bhvN30nD8unnXCJ2MvV/BlVluYOva6NbYT+ptKuhl8nRRljjRgVqtRt51IMjLBUoLgxXgmL0fjliTDv+qT0RE9sRw5YiKByvAcJKLoq+JHIhWK+JGcibOFwSp6DtSmErKKH12uZI8HxmChn7uhTdJLLj5YeGNESHdaLHgeiGF7qaLxW6uWPRmi7KCmyvqp9AuuCHjiRv3MXbZ8TJrGta+rs1PlB2x98MRayIiIrI3hitHpNWYnrxC91qrsX1NVCVotCKOXr+HE0kCal2/90BDpnLUGlyMSy8SolJxIS4dWXnG309BAEJ93NCstifCa3ugib87Jq87i8T03FKv25nSt6nNTta7NPJ1uNndinLE3g9HrImIiMieGK4cUWk3CGaPVaXgiLMBGc7GJcfPl/82ezau+5l5BUP6UvXD+64mZkJj4qIllVKGJgEeCK/tgWa1PRAe6IHGAe5wdTL8z837TzRzqOt2HPlaIiIiIqocGK6IrMwRpxQuaYrxuNQcvLTipH5KaFEUcet+tkGIOn8nrcTJJ7zdnKQAVRCimtX2QIhPDbMCiCNet+OINREREVHlwXBFZEXmhhhbKmvac0CaYnzxgeu4EJeO9Jx8k8epX8tVH6KkXilP+Lk7lziRgzkc8bodR6yJiIiIKgeGKyIrMSfETN94Ho39PSCicHpw3bTg+fqfWoOpw/MN1muRrykytXjBVOQG67WGU5HfvJdV6rTngDTF+PEb0sx8TnIZGgXUKOiJKrhGKsAd7qqKuYGfI16344g1ERERkeNjuCKykrLu3QQACem56D53j20KstAzEXUx8uF6CPOrAaVcZu9yiIiIiCodhisiK0lILz1Y6TjLBTgr5QVTg8ugKJgiXCEv+Fl0uvCi64tMLV64TFa4j8Hygn3lAuJSc7DpzJ0y63qsZW00DfR40GYgIiIiqrYYroisICkjF3+eu2vWtsvGRdh0yJlGK+L4jXsOO8U4ERERUVXBsT9EDyAtR4152y+iy5zd2HouvtRtBUizBto6xOimGNfVULwmgFOMExEREVkDwxVROeSoNfhh31V0mbMbC3ZdQVaeBi2CPPHqo2EQ4HghRjfFeICnymB5gKfKLjMYEhEREVVFHBZIFnHEm+Paklqjxa9/38KCnZcRlyZdY9XA1w1v9W6MPs0DIAgCwgM9HPI+Sbopxg9fScD2/UfROzICHcP8qtXvj4iIiKgiMVyR2Rzx5ri2otWK+N8/dzFv+0XcSM4CAAR5ueC1ng3xZJsgKIrMrufI90mSywREhHgjOUZEhIPURERERFRVMFyRWRzx5ri2IIoidl9MwGfbLiHmbhoAoJabE17pHoZnHq4LZ4Xc5H68TxIRERFR9cNwRWUq6+a4AoBZm6PRKzygSvWEHLt+D59tu6C/ua67swLPdwnFuM4hqOHMfzpEREREZIhniFSmsm6OKwK4m5qDY9fvVYnemvN3UvHZtovYczERAOCskGF0p/p4qWsD1HRzsnN1REREROSoGK6oTObeHPdk7H1EhHhDVkl7r64nZWLu9ov431npflVymYCh7YLx2qMNjWbZIyIiIiIqjuGKyuTnbl6w+GzbRSw5cB2dG/qgc5gPIhv6VopQcjc1Gwt2XsYvf9+CRisNfhzQqjYm9WqEEB83O1dHRERERJUFwxWVqUOINwI9VYhLzTF53RUgDZ2TCUByZh42nr6DjafvAAAa+tVA54Y+iGzog4iQWnBzoGuV7mfm4Zs9V/DT4X+Rl68FAHRv7Iu3ohqjWW1PO1dHRERERJWN45zpksOSywTMGBCOl1acNFqnGwD45fDW6NHEHydj7+PA5STsv5KEs7dScDkhA5cTMrD04A0o5QIeqlsTkQ190LmhL1oE2SfAZOTmY/H+61i0/xoycvMBAO3r18TkPk3Qvr63XWoiIiIiosqP4YrM0qd5IL4Y1hqvrz1tsLz4zXEfDq2Fh0Nr4a2oxkjJysOhq8nYfzkJ+y8n4tb9bBy9fg9Hr9/D59svwdNFiY6h3vDMFtDifhZC/So2bOWoNVh5NBbf7L6C5Mw8AEDTQA9MjmqMbo19IQiV81oxIiIiInIMDFdkNmeFdKNcfw9nvNO3Kfw8Sr85rperE/q1CES/FoEQRRH/Jmdh/5UkHLiciENXkpGarcbW8/EA5Fg77wDq13ItGELoi44NasFDpbRK3fkaLdafvI35f13CnYJZD0N83DCpVyP0bxFYaSfgICIiIiLHwnBFZtseHQ8AeLxVbTzRJsiifQVBQH0fN9T3ccOzD9dDvkaLM7dSsfdiPDYfv4LYTBluJGfhRnIsVhyJhVwmoFUdT3Ru6IsuDX3QKtgLSrnM5LE1WhHHrt9DQnoO/NwLA58oivjzXBzmbr+Iq4mZAIAADxVe69kQT7WtU+LxiIiIiIjKg+GKzKLWaLEzRgpXUc0CHvh4CrkMbevVRMvaNdAg+yIiezyKEzfTcOByIvZfTsK1pEycjE3BydgULNh5GTWcFXg4tBa6NJJmIgzxcYMgCNh67i5mbY42uA9XgKcKQ9rWwZ6LifjndioAoKarEi93C8OzHetBpZQ/cP1ERERERMUxXJFZjl67h7ScfPjUcEKbujWtfnx3lQK9wv3RK9wfAHA7JVsftA5eScL9LDX+ionHXwUBL8jLBfVrueLg1WSjY8Wl5uCrXVcAAG5OcoyPDMXzkSFwt9IwQyIiIiIiUxiuyCzbo+MAAD2b+pd4jZU1BXm5YFj7uhjWvi60WhHn76Rh/5VE7L+UhBP/3sftlGzcTsku9RhuTnLseqsb/D0c/15bRERERFT5MVxRmURRxPbzUo9R72b+Nn9/mUxAizqeaFHHEy93C0NWXj5+OnQDn269WOp+mXkaXEvMZLgiIiIiIpvgFf1Upn9upyIuLQduTnJ0auBj73Lg6qRAbS8Xs7ZNSM8peyMiIiIiIitguKIybTsvDQns1tjPYSaD8HM3rzfK3O2IiIiIiB4UwxWVyZ5DAkvSIcQbgZ4qlHT1lwAg0FOalp2IiIiIyBYYrqhU1xIzcDkhAwqZgG6N/exdjp5cJmDGgHAAMApYutczBoTbZPINIiIiIiKA4YrKoLtxcMcGteDp4lhTmfdpHohvRz6EAE/DoX8Bnip8O/Ih9GkeaKfKiIiIiKg64myBVKrtBddb9bbCjYMrQp/mgegVHoBj1+8hIT0Hfu7SUED2WBERERGRrTFcUYkS0nJw6mYKAKBXU8e53qo4uUxAxwa17F0GEREREVVzHBZIJforJgGiCLQK9jIaekdERERERIYYrqhEuinYoxxolkAiIiIiIkfFcEUmpeeocehqEgCgd7hjXm9FRERERORIGK7IpD0XE6HWiAj1dUOYXw17l0NERERE5PAYrsgk3RTs7LUiIiIiIjIPwxUZyc3XYPeFBAC83oqIiIiIyFwMV2Tk8NVkZOTmw8/dGa3qeNm7HCIiIiKiSoHhiozohgT2CveHjDfjJSIiIiIyC8MVGdBqRezQXW/VjNdbERERERGZi+GKDJy6mYLE9Fy4OyvQMbSWvcshIiIiIqo0GK7IwPZo6cbB3Zv4wUnBrwcRERERkbl49kx6oihi+3ndkEDOEkhEREREZAmGK9K7mpiB60mZcJLL0LWRr73LISIiIiKqVBiuSG9bQa/VI2G14K5S2rkaIiIiIqLKheGK9Lafl6634iyBRERERESWY7giAMDd1GycuZUKQQAebepn73KIiIiIiCodhisCAPxVcG+rh+rWhJ+7ys7VEBERERFVPgxXBKDweqsozhJIRERERFQuDFeE1Cw1jlxLBgD0Cuf1VkRERERE5cFwRdh9MQH5WhGN/GsgxMfN3uUQEREREVVKDFeE7dEFswSy14qIiIiIqNwYrqq5HLUGey4mAgCiOAU7EREREVG5MVxVcwevJCErT4NATxWaB3nYuxwiIiIiokqL4aqa214wS2DvcH8IgmDnaoiIiIiIKi+Gq2pMoxXxV0xBuOKQQCIiIiKiB8JwVY2d+Pc+kjPz4OmiRIcQb3uXQ0RERERUqTFcVWPbz0uzBD7axA9KOb8KREREREQPgmfU1ZQoitgerRsS6G/naoiIiIiIKj+Gq2rqYnw6Yu9lwVkhQ5dGvvYuh4iIiIio0mO4qqa2nZN6rSIb+sLVSWHnaoiIiIiIKj+Gq2pqe7R0vRWHBBIRERERWQfDVTV0634Wzt9Jg0yQJrMgIiIiIqIHZ/dwtXDhQtSvXx8qlQoRERE4duxYqdvPnz8fjRs3houLC4KDg/HGG28gJydHv37mzJkQBMHg0aRJk4r+GJXKjoKJLNrV90atGs52roaIiIiIqGqw68U2a9euxaRJk/Ddd98hIiIC8+fPR1RUFC5evAg/P+MelVWrVmHKlClYsmQJOnXqhEuXLmHMmDEQBAHz5s3Tb9esWTP89ddf+tcKBa8pKmpbwRTsvcM5JJCIiIiIyFrs2nM1b948PP/88xg7dizCw8Px3XffwdXVFUuWLDG5/aFDh/DII4/g6aefRv369dG7d2+MGDHCqLdLoVAgICBA//Dx8bHFx6kU7mfm4dj1ewCAqGYBdq6GiIiIiKjqsFuXTl5eHk6cOIGpU6fql8lkMvTs2ROHDx82uU+nTp2wYsUKHDt2DB06dMC1a9ewZcsWPPvsswbbXb58GbVr14ZKpULHjh0xe/Zs1K1bt8RacnNzkZubq3+dlpYGAFCr1VCr1Q/yMR3O9vN3oBWBJgHuCHBX2v3z6d7f3nVUJ2xz22J72x7b3PbY5rbF9rY9trntOVKbW1KDIIqiWIG1lOjOnTsICgrCoUOH0LFjR/3yyZMnY+/evTh69KjJ/RYsWIC33noLoigiPz8fL774Ir799lv9+j///BMZGRlo3Lgx7t69i1mzZuH27ds4d+4c3N3dTR5z5syZmDVrltHyVatWwdXV9QE/qWP58YIM/9yXoU8dLfoGa+1dDhERERGRQ8vKysLTTz+N1NRUeHh4lLptpboYac+ePfj444/xzTffICIiAleuXMFrr72GDz74AO+99x4AoG/fvvrtW7ZsiYiICNSrVw+//PILxo8fb/K4U6dOxaRJk/Sv09LSEBwcjN69e5fZgJVJdp4Gb/+9G4AWLz3eCeGB9v9sarUaO3bsQK9evaBUKu1dTrXANrcttrftsc1tj21uW2xv22Ob254jtbluVJs57BaufHx8IJfLER8fb7A8Pj4eAQGmrwV677338Oyzz+K5554DALRo0QKZmZmYMGEC3n33XchkxpeQeXl5oVGjRrhy5UqJtTg7O8PZ2XjWPKVSafdfpjXtupSMHLUWdWq6oGWwNwRBsHdJelWtrSsDtrltsb1tj21ue2xz22J72x7b3PYcoc0teX+7TWjh5OSEtm3bYufOnfplWq0WO3fuNBgmWFRWVpZRgJLL5QCAkkY3ZmRk4OrVqwgMDLRS5ZXX9vNSkO0dHuBQwYqIiIiIqCqw67DASZMmYfTo0WjXrh06dOiA+fPnIzMzE2PHjgUAjBo1CkFBQZg9ezYAYMCAAZg3bx7atGmjHxb43nvvYcCAAfqQ9dZbb2HAgAGoV68e7ty5gxkzZkAul2PEiBF2+5yOIF+jxc4LBeGqGadgJyIiIiKyNruGq2HDhiExMRHTp09HXFwcWrduja1bt8LfXzr5j42NNeipmjZtGgRBwLRp03D79m34+vpiwIAB+Oijj/Tb3Lp1CyNGjEBycjJ8fX3RuXNnHDlyBL6+vjb/fI7k2I17SMlSo6arEu3q1bR3OUREREREVY7dJ7SYOHEiJk6caHLdnj17DF4rFArMmDEDM2bMKPF4a9assWZ5VYZuSGDPpv5QyO16ezMiIiIioiqJZ9nVgCiK2BGtGxLIGwcTEREREVUEhqtq4PydNNxOyYaLUo7Ihj72LoeIiIiIqEpiuKoGtp+PAwB0aeQDlVJu52qIiIiIiKomhqtqYHvBkMAoDgkkIiIiIqowDFdV3L/JmbgQlw65TECPJn72LoeIiIiIqMpiuKridBNZRIR4w8vVyc7VEBERERFVXQxXVdy2guuteofzxsFERERERBXJ7ve5ooqTlJGLv/+9D4BTsBMREZHj0Gg0UKvV9i7DbGq1GgqFAjk5OdBoNPYup1qwZZsrlUrI5daZ9I3hqgrbGRMPUQRaBHmitpeLvcshIiKiak4URcTFxSElJcXepVhEFEUEBATg5s2bEATB3uVUC7Zucy8vLwQEBDzwezFcVWHbzxfcOJhDAomIiMgB6IKVn58fXF1dK01Q0Wq1yMjIQI0aNSCT8aoaW7BVm4uiiKysLCQkJAAAAgMDH+h4DFdVVEZuPvZfSQLAIYFERERkfxqNRh+satWqZe9yLKLVapGXlweVSsVwZSO2bHMXF2mEV0JCAvz8/B5oiCC/HVXUvkuJyMvXon4tVzTyr2HvcoiIiKia011j5erqaudKiIzpvpcPei0gw1UVtV03S2CzBx87SkRERGQtPC8hR2St76XF4ap+/fp4//33ERsba5UCyPrUGi12XpDGjfJ6KyIiIiIi27A4XL3++utYv349QkND0atXL6xZswa5ubkVURuV05FryUjPyYdPDSe0qVvT3uUQERERkQ1069YNr7/+uv51/fr1MX/+fLvVUx2VK1ydPn0ax44dQ9OmTfGf//wHgYGBmDhxIk6ePFkRNZKFdLME9gr3h1zGrnciIiKqWjRaEYevJmPj6ds4fDUZGq1Y4e9569YtjB8/HrVr14aTkxPq1auH1157DcnJyRX+3hXt1q1bcHJyQvPmze1dSqVX7muuHnroISxYsAB37tzBjBkz8OOPP6J9+/Zo3bo1lixZAlGs+C85GdNqReyI1k3BzlkCiYiIqGrZeu4uOn+6CyMWHcFra05jxKIj6PzpLmw9d7fC3vPatWvo0aMHrly5gtWrV+PKlSv47rvvsHPnTnTs2BH37t2rsPcGHnyShbIsW7YMQ4cORVpaGo4ePVqh71UWjUYDrVZr1xoeRLnDlVqtxi+//ILHH38cb775Jtq1a4cff/wRgwcPxjvvvINnnnnGmnWSmf65nYq4tBy4OcnRsUHlmuaUiIiIqDRbz93FSytO4m5qjsHyuNQcvLTiZIUFrIkTJ0KpVGLr1q3o2rUr6tati759++Kvv/7C7du38e677wIA3nnnHURERBjt36pVK7z//vv61z/++COaNm0KlUqFJk2a4JtvvtGvu3HjBgRBwNq1a9G1a1eoVCqsXLkSycnJGDFiBIKCguDq6ooWLVpg9erVD/zZRFHE0qVL8eyzz+Lpp5/G4sWLjbY5ePAgunXrBldXV9SsWRNRUVG4f/8+AGnK9Dlz5iAsLAzOzs6oW7cuPvroIwDAnj17IAiCwU2jT58+DUEQcOPGDQBSsPPy8sKmTZsQHh4OZ2dnxMbG4vjx4xg0aBD8/Pzg6emJrl27Go2SS0lJwQsvvAB/f3+oVCo0b94c//vf/5CZmQkPDw/89ttvBttv2LABbm5uSE9Pf+B2K4nF97k6efIkli5ditWrV0Mmk2HUqFH44osv0KRJE/02gwYNQvv27a1aKJlnW8Esgd0a+0GlLP8c/UREREQVTRRFZKs1Zm2r0YqYsek8TI2NEgEIAGZuisYjYT5mXRbhopSbNUPcvXv3sH37dkybNk1/PySdgIAAPPPMM1i7di2++eYbPPPMM5g9ezauXr2KBg0aAADOnz+Ps2fPYt26dQCAlStXYvr06fj666/Rpk0bnDp1Cs8//zzc3NwwevRo/bGnTJmCuXPnok2bNlCpVMjJyUHbtm3x9ttvw8PDA3/88QeeffZZNGjQAB06dCjzc5Rk9+7dyMrKQs+ePREUFIROnTrhiy++gJubGwApDD366KMYN24cvvzySygUCuzevRsajfR7mzp1KhYtWoQvvvgCnTt3xt27d3HhwgWLasjKysKnn36KH3/8EbVq1YKfnx+uXLmC4cOHY+HChRAEAXPnzkW/fv1w+fJluLu7Q6vVom/fvkhPT8eKFSvQoEEDREdHQy6Xw83NDcOHD8fSpUvx1FNP6d9H99rd3b3c7VUWi8NV+/bt0atXL3z77bcYOHAglEql0TYhISEYPny4VQoky2zXDQlsxlkCiYiIyLFlqzUIn77NKscSAcSl5aDFzO1mbR/9fhRcnco+Fb58+TJEUUTjxo1Nrm/atCnu37+PxMRENGvWDK1atcKqVavw3nvvAZDCVEREBMLCwgAAM2bMwNy5c/Hkk08CkM6bo6Oj8f333xuEq9dff12/jc5bb72lf/6f//wH27Ztwy+//PJA4Wrx4sUYPnw45HI5mjdvjtDQUPz6668YM2YMAGDOnDlo166dQe9as2bNAADp6en48ssv8fXXX+trb9CgATp37mxRDWq1Gt988w1atWqlX9ajRw+0a9cOHh4ekMlk+OGHH+Dl5YW9e/fisccew19//YVjx44hJiYGjRo1AgCEhobq93/uuefQqVMn3L17F4GBgUhISMCWLVvw119/laudzGXxsMBr165h69atGDJkiMlgBQBubm5YunTpAxdHlrmamIErCRlQygV0b+Jn73KIiIiIqgxz5xN45plnsGrVKv0+q1ev1l8uk5mZiatXr2L8+PGoUaOG/vHhhx/i6tWrBsdp166dwWuNRoMPPvgALVq0gLe3N2rUqIFt27Y90O2RUlJSsH79eowcOVK/bOTIkQZDA3U9V6bExMQgNze3xPXmcnJyQsuWLQ2WxcfH47XXXkPjxo3h6ekJDw8PZGRk6D/v6dOnUadOHX2wKq5Dhw5o1qwZfvrpJwDAihUrUK9ePXTp0uWBai2LxT1XCQkJiIuLMxpPevToUcjlcqMvAtmObiKLh0NrwUNlOvgSEREROQoXpRzR70eZte2x6/cwZunxMrdbNrY9OoR4m/Xe5ggLC4MgCLh48aLJ9TExMahZsyZ8fX0BACNGjMDbb7+NkydPIjs7Gzdv3sSwYcMAABkZGQCARYsWGZ1Ly+WG9eiG5el89tln+PLLLzF//ny0aNECbm5ueP3115GXl2fW5zBl1apVyMnJMahFFEVotVpcunQJjRo1MhoKWVRp6wBAJpPpj6ljanIOFxcXoyGaY8aMQWJiIr744guEhITA2dkZHTt21H/est4bkHqvFi5ciClTpmDp0qUYO3Zshd/E2uKeq1deeQU3b940Wn779m288sorVimKykd3vVXvZpwlkIiIiByfIAhwdVKY9Yhs6ItATxVKOjUWAAR6qhDZ0Nes45l7kl2rVi307NkTS5YsQXZ2tsG6uLg4rFy5EsOGDdMfr06dOujatStWrlyJlStXolevXvDzk0YU+fv7o3bt2rh27RrCwsIMHiEhIaXWcfDgQTzxxBMYOXIkWrVqhdDQUFy6dMmsz1CSxYsX480338Tp06f1jzNnziAyMhJLliwBALRs2RI7d+40uX/Dhg3h4uJS4npd4Lx7t3CikdOnT5tV26FDhzBhwgT069cPzZo1g7OzM5KSkvTrW7ZsiVu3bpXaBiNHjsS///6LBQsWIDo62mDYZUWxOFxFR0fjoYceMlrepk0bREdHW6UoslxCWg5OxaYAAHqH83orIiIiqlrkMgEzBoQDgFHA0r2eMSC8Qu7x+dVXXyE3Nxd9+/bFvn37cPPmTWzduhW9evVCUFCQfnY8nWeeeQZr1qzBr7/+ajSD9qxZszB79mwsWLAAly5dwj///IOlS5di3rx5pdbQsGFD7NixA4cOHUJMTAxeeOEFxMfHl/sznT59GidPnsRzzz2H5s2bGzxGjBiBn376Cfn5+Zg6dSqOHz+Ol19+GWfPnsWFCxfw7bffIikpCSqVCm+//TYmT56Mn3/+GVevXsWRI0f0wwrDwsIQHByMmTNn4vLly/jjjz8wd+5cs+pr2LAhfvnlF8TExODo0aN45plnDHqrunbtii5dumDw4MHYsWMHrl+/jj///BNbt27Vb1OzZk08+eST+O9//4vevXujTp065W4vc1kcrpydnU3+Iu/evQuFwuJRhmQlO2Kk30nrYC/4e6jsXA0RERGR9fVpHohvRz6EAE/Dc50ATxW+HfkQ+jQPrJD3bdiwIXbt2oWQkBAMHToUDRo0wIQJE9C9e3ccPnwY3t6GwxCfeuopJCcnIysrCwMHDjRY99xzz+HHH3/E0qVL0aJFC3Tt2hXLli0rs+dq2rRpeOihhxAVFYVu3bohICDA6NiWWLx4McLDww1m/NYZNGiQfgKIRo0aYfv27Thz5gw6dOiAjh07YuPGjfrz/vfeew9vvvkmpk+fjqZNm2LYsGFISEgAACiVSqxevRoXLlxAy5Yt8emnn+LDDz80q75FixYhJSUF7dq1w7PPPotXX31V3wOos27dOrRv3x4jRoxAeHg4Jk+erJ/FUGf8+PHIy8vDuHHjytNMFhNEC+/2O2LECNy9excbN26Ep6cnAOliuIEDB8LPzw+//PJLhRRqS2lpafD09ERqaio8PDzsXY5ZRi85hr2XEjG5T2O83C3M3uWYTa1WY8uWLejXr1+JE6SQdbHNbYvtbXtsc9tjm9tWZW3vnJwcXL9+HSEhIVCpHuwPwRqtiGPX7yEhPQd+7ip0CPGukB4rHa1Wi7S0NP3MdVTxrNXmy5cvxxtvvIE7d+7AycmpxO1K+35akg0s7mr6/PPP0aVLF9SrVw9t2rQBIHUr+vv7Y/ny5ZYejqwgLUeNQ1elMai9w3m9FREREVVtcpmAjg1q2bsMcmBZWVm4e/cuPvnkE7zwwgulBitrsjgGBgUF4ezZs5gzZw7Cw8PRtm1bfPnll/jnn38QHBxcETVSGfZcTIRaI6KBrxvC/GrYuxwiIiIiIruaM2cOmjRpgoCAAEydOtVm71uui6Tc3NwwYcIEa9dC5bSdswQSEREREenNnDkTM2fOtPn7lnsGiujoaMTGxhrNrf/4448/cFFkvtx8DfZcTATAWQKJiIiIiOzJ4nB17do1DBo0CP/88w8EQdDfFEw3t3/xGTqoYh26moyM3Hz4uTujVR0ve5dDRERERFRtWXzN1WuvvYaQkBAkJCTA1dUV58+fx759+9CuXTvs2bOnAkqk0mw/L03B3ruZP2QVOEsOERERERGVzuKeq8OHD2PXrl3w8fGBTCaDTCZD586dMXv2bLz66qs4depURdRJJmi1InZEF4QrzhJIRERERGRXFvdcaTQauLu7AwB8fHxw584dAEC9evVw8eJF61ZHpTp1MwVJGblwd1bg4VBOR0pEREREZE8W91w1b94cZ86cQUhICCIiIjBnzhw4OTnhhx9+QGhoaEXUSCXQzRLYvYkfnBS8oR0RERERkT1ZfEY+bdo0aLVaAMD777+P69evIzIyElu2bMGCBQusXiCZJooithWEqyhOwU5ERETkMLp164bXX3/d3mWQHVgcrqKiovDkk08CAMLCwnDhwgUkJSUhISEBPXr0sHqBZNqVhAzcSM6Ck1yGro197V0OERERUZU1duxY1KxZEy+99JLRuldeeQWCIGDMmDH6ZevXr8cHH3xgwwoNrV69GnK5HK+88ordaqiuLApXarUaCoUC586dM1ju7e2tn4qdbEPXa/VIWC3UcC737cqIiIiIKo/ds4G9c0yv2ztHWl9BgoKCsHbtWmRnZ+uX5eTkYNWqVahbt67Btt7e3vo5CiwliiLy8/MfqNbFixdj8uTJWL16NXJych7oWA+q+D1xqzqLwpVSqUTdunV5LysHsF03SyCHBBIREVF1IZMDuz8yDlh750jLZfIKe+tWrVohODgY69ev1y9bv3496tatizZt2hhsW3xYYG5uLt5++20EBwfD2dkZYWFhWLx4MQBgz549EAQBf/75J9q2bQtnZ2ccOHAAubm5ePXVV+Hn5weVSoXOnTvj+PHjZdZ5/fp1HDp0CFOmTEGjRo0M6tVZsmQJmjVrBmdnZwQGBmLixIn6dSkpKXjhhRfg7+8PlUqF5s2b43//+x8AYObMmWjdurXBsebPn4/69evrX48ZMwYDBw7ERx99hNq1a6Nx48YAgOXLl6Ndu3Zwd3dHQEAAnn76aSQkJBgc6/z583jsscfg4eEBT09P9O3bF1evXsW+ffugVCoRFxdnsP3rr7+OyMjIMtvEliweFvjuu+/inXfewb179yqiHjLDnZRsnL2VCkEAejb1t3c5REREROUjikBepvmPjq8AXf4rBaldH0rLdn0ove7yX2m9uccSRYvLHTt2LJYuXap/vWTJEowdO7bM/UaNGoXVq1djwYIFiImJwffff48aNWoYbDNlyhR88skniImJQcuWLTF58mSsW7cOP/30E06ePImwsDBERUWVeQ6+dOlS9O/fH56enhg5cqQ+xOl8++23eOWVVzBhwgT8888/2LRpE8LCwgAAWq0Wffv2xcGDB7FixQpER0fjk08+gVxuWWjduXMnLl68iB07duiDmVqtxgcf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GK5spLK2OQeNEhERERERWQHDFRERERERkRUwXBEREREREVkBwxUREREREZEVMFwRERERERFZAcMVERERERGRFTBcERERERERWQHDFRERERERkRUwXBEREREREVkBwxUREREREZEVMFwRERERERFZAcMVERERERGRFTBcERERERERWYHdw9XChQtRv359qFQqRERE4NixY6VuP3/+fDRu3BguLi4IDg7GG2+8gZycnAc6JhERERER0YOya7hau3YtJk2ahBkzZuDkyZNo1aoVoqKikJCQYHL7VatWYcqUKZgxYwZiYmKwePFirF27Fu+88065j0lERERERGQNdg1X8+bNw/PPP4+xY8ciPDwc3333HVxdXbFkyRKT2x86dAiPPPIInn76adSvXx+9e/fGiBEjDHqmLD0mERERERGRNSjs9cZ5eXk4ceIEpk6dql8mk8nQs2dPHD582OQ+nTp1wooVK3Ds2DF06NAB165dw5YtW/Dss8+W+5gAkJubi9zcXP3rtLQ0AIBarYZarX6gz0ml07Uv29l22Oa2xfa2Pba57bHNbYvtbXtsc9tzpDa3pAa7haukpCRoNBr4+/sbLPf398eFCxdM7vP0008jKSkJnTt3hiiKyM/Px4svvqgfFlieYwLA7NmzMWvWLKPl27dvh6urq6Ufjcphx44d9i6h2mGb2xbb2/bY5rbHNrcttrftsc1tzxHaPCsry+xt7RauymPPnj34+OOP8c033yAiIgJXrlzBa6+9hg8++ADvvfdeuY87depUTJo0Sf86LS0NwcHB6N27Nzw8PKxROpVArVZjx44d6NWrF5RKpb3LqRbY5rbF9rY9trntsc1ti+1te2xz23OkNteNajOH3cKVj48P5HI54uPjDZbHx8cjICDA5D7vvfcenn32WTz33HMAgBYtWiAzMxMTJkzAu+++W65jAoCzszOcnZ2NliuVSrv/MqsLtrXtsc1ti+1te2xz22Ob2xbb2/bY5rbnCG1uyfvbbUILJycntG3bFjt37tQv02q12LlzJzp27Ghyn6ysLMhkhiXL5XIAgCiK5TomERERERGRNdh1WOCkSZMwevRotGvXDh06dMD8+fORmZmJsWPHAgBGjRqFoKAgzJ49GwAwYMAAzJs3D23atNEPC3zvvfcwYMAAfcgq65hEREREREQVwa7hatiwYUhMTMT06dMR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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Random Forest Classification"
      ],
      "metadata": {
        "id": "KyEFrTjX1Dh_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Delineate hyperparameters\n",
        "N_ESTIMATORS = 100    # Number of trees in the forest\n",
        "MAX_DEPTH = None      # Maximum depth of the trees\n",
        "MIN_SAMPLES_SPLIT = 2 # Minimum number of samples required to split an internal node\n",
        "MIN_SAMPLES_LEAF = 1  # Minimum number of samples required to be at a leaf node\n",
        "RANDOM_STATE = 42     # Random state for reproducibility"
      ],
      "metadata": {
        "id": "vVP675Ek1CWL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)"
      ],
      "metadata": {
        "id": "2id_-BelRpBT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize and train the Random Forest model\n",
        "\n",
        "X_train_pca = pca.fit_transform(X_train)\n",
        "X_test_pca = pca.transform(X_test)\n",
        "\n",
        "# Build up random forest Model.\n",
        "# n_jobs = -1 put all available CPU in the computer to fast the classification Process.\n",
        "clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,\n",
        "                             max_depth=MAX_DEPTH,\n",
        "                             min_samples_split=MIN_SAMPLES_SPLIT,\n",
        "                             min_samples_leaf=MIN_SAMPLES_LEAF,\n",
        "                             random_state=RANDOM_STATE)\n",
        "clf.fit(X_train_pca, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "IE0L_S-YPu3v",
        "outputId": "afe62720-5b7d-4af0-b09c-b5cd05ee32f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier(random_state=42)"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 96
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Predict using the test set\n",
        "y_pred = clf.predict(X_test_pca)"
      ],
      "metadata": {
        "id": "eqIY9VZwP3B2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### RF Multi-Class performance measurement"
      ],
      "metadata": {
        "id": "7TegQaAGwzWL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "NUM_CLASSES = len(set(y_train))\n",
        "\n",
        "conf_mat = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]\n",
        "err_cnt = 0\n",
        "# Use y_test for true labels, so the row represents the actual class.\n",
        "for i, val in enumerate(y_test):\n",
        "  # Use y_pred for predicted labels, the column represent the predicted class.\n",
        "  conf_mat[val][y_pred[i]] += 1\n",
        "\n",
        "print(\"Confusion matrix:\")\n",
        "for i in conf_mat:\n",
        "  print(i)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jUZzcfP1P_T6",
        "outputId": "7132bff9-899d-4dbb-fec0-a8ef90d1d09a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Confusion matrix:\n",
            "[44461, 184, 848, 59]\n",
            "[1211, 9, 47, 4]\n",
            "[3506, 51, 240, 27]\n",
            "[503, 7, 54, 8]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "c_mat = [[ 8,  5,  20],\n",
        "         [ 2, 10,  10],\n",
        "         [ 5,  5, 270]]\n",
        "\n",
        "\n",
        "c_mat = np.array(conf_mat)\n",
        "\n",
        "#Determine overall accuracy\n",
        "total_cnt = 0\n",
        "total_correct = 0\n",
        "for i in range(NUM_CLASSES):\n",
        "  for j in range(NUM_CLASSES):\n",
        "    total_cnt += c_mat[i][j]\n",
        "    if i==j:\n",
        "      total_correct += c_mat[i][j]\n",
        "overall_acc = total_correct/total_cnt\n",
        "print(\"total count:\", total_cnt)\n",
        "print(\"total correct:\", total_correct)\n",
        "print(\"overall accuracy %3.3f\" % overall_acc)\n",
        "\n",
        "print(\"\\nConfusion matrix:\")\n",
        "for row in c_mat:\n",
        "      print(row)\n",
        "\n",
        "tp = TP(c_mat)\n",
        "fp = FP(c_mat)\n",
        "tn = TN(c_mat)\n",
        "fn = FN(c_mat)\n",
        "acc = accuracy(tp, tn, fp, fn)\n",
        "acc = [round(x, 3) for x in acc]\n",
        "tpr = TPR(tp, fn)\n",
        "tpr = [round(x, 3) for x in tpr]\n",
        "f1 = F1(tp, fp, fn)\n",
        "f1 = [round(x, 3) for x in f1]\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YMIhUs1gQC-j",
        "outputId": "eb7c131d-93e5-425a-cc20-fc44f03a4822"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "total count: 51219\n",
            "total correct: 44718\n",
            "overall accuracy 0.873\n",
            "\n",
            "Confusion matrix:\n",
            "[44461   184   848    59]\n",
            "[1211    9   47    4]\n",
            "[3506   51  240   27]\n",
            "[503   7  54   8]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "macro_tpr = macro_TPR(tpr, c_mat)\n",
        "macro_f1 = macro_F1(f1)\n",
        "macro_acc = macro_ACCURACY(acc)\n",
        "\n",
        "micro_c_mat = micro_conf_mat(tp, tn, fp, fn)\n",
        "\n",
        "\n",
        "for i in range(2):\n",
        "  for j in range(2):\n",
        "    micro_c_mat[i][j] = round(micro_c_mat[i][j], 3)\n",
        "\n",
        "micro_tpr = micro_TPR(micro_c_mat)\n",
        "micro_f1 = micro_F1(micro_c_mat)\n",
        "micro_acc = micro_ACCURACY(micro_c_mat)"
      ],
      "metadata": {
        "id": "DPeKPpapQFTe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"\\nClass metrics.\")\n",
        "print(\"TP:\", tp)\n",
        "print(\"FP:\", fp)\n",
        "print(\"TN:\", tn)\n",
        "print(\"FN:\", fn)\n",
        "print(\"Accuracy:\", acc)\n",
        "print(\"TPR:\", tpr)\n",
        "print(\"F1:\", f1)\n",
        "\n",
        "print(\"\\nMacro metrics.\")\n",
        "print(\"Macro TPR: %2.3f\" % macro_tpr)\n",
        "print(\"Macro F1: %2.3f\" % macro_f1)\n",
        "print(\"Macro ACC: %2.3f\" % macro_acc)\n",
        "\n",
        "print(\"\\nClass confusion matrices:\")\n",
        "for i in range(len(tp)):\n",
        "  class_c_mat = class_conf_mat(i, tp, tn, fp, fn)\n",
        "  print(\"Class \", i)\n",
        "  print(class_c_mat)\n",
        "\n",
        "print(\"\\nMicro confusion matrix:\")\n",
        "for row in micro_c_mat:\n",
        "  print(row)\n",
        "print(\"\\nMicro metrics.\")\n",
        "print(\"Micro TPR: %2.3f\" % micro_tpr)\n",
        "print(\"Micro F1: %2.3f\" % micro_f1)\n",
        "print(\"Micro ACC: %2.3f\" % micro_acc)"
      ],
      "metadata": {
        "id": "UxfhVM2ZR4AY",
        "outputId": "e29d18a1-cde3-4aa2-e90a-8800002cdd88",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Class metrics.\n",
            "TP: [44461, 9, 240, 8]\n",
            "FP: [5220, 242, 949, 90]\n",
            "TN: [-44014, 49697, 46206, 50549]\n",
            "FN: [1091, 1262, 3584, 564]\n",
            "Accuracy: [0.066, 0.971, 0.911, 0.987]\n",
            "TPR: [0.976, 0.007, 0.063, 0.014]\n",
            "F1: [0.934, 0.012, 0.096, 0.024]\n",
            "\n",
            "Macro metrics.\n",
            "Macro TPR: 0.000\n",
            "Macro F1: 0.267\n",
            "Macro ACC: 0.734\n",
            "\n",
            "Class confusion matrices:\n",
            "Class  0\n",
            "[[44461, 5220], [1091, -44014]]\n",
            "Class  1\n",
            "[[9, 242], [1262, 49697]]\n",
            "Class  2\n",
            "[[240, 949], [3584, 46206]]\n",
            "Class  3\n",
            "[[8, 90], [564, 50549]]\n",
            "\n",
            "Micro confusion matrix:\n",
            "[44718, 6501]\n",
            "[6501, 102438]\n",
            "\n",
            "Micro metrics.\n",
            "Micro TPR: 0.919\n",
            "Micro F1: 0.919\n",
            "Micro ACC: 0.919\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### Micro ACC = 0.919, Macro ACC = 0.734.\n",
        "It is not a huge difference between random forest and KNN. And Micro statistics is still better performed."
      ],
      "metadata": {
        "id": "ywAI4UwuEEk5"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Testing with different Parameters.\n",
        "These two parameters N_ESTIMATORS and MAX_DEPTH usually have the most direct impact on the accuracy. So, I will test on the accuracy change on these two parameters."
      ],
      "metadata": {
        "id": "KZu9ZbE7D84L"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "depth_values = range(1, 21)\n",
        "\n",
        "# Store accuracies for each K value\n",
        "\n",
        "macro_accuracies = []\n",
        "micro_accuracies = []\n",
        "overall_accuracies = []\n",
        "\n",
        "for depth in depth_values:\n",
        "    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)\n",
        "    rf.fit(X_train_pca, y_train)\n",
        "    y_pred = rf.predict(X_test_pca)\n",
        "\n",
        "    # ----* calcualte macro accuracy *---- #.\n",
        "    NUM_CLASSES = len(set(y_train))\n",
        "\n",
        "    conf_mat = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]\n",
        "    err_cnt = 0\n",
        "\n",
        "    # Use y_test for true labels, so the row represents the actual class.\n",
        "    for i, val in enumerate(y_test):\n",
        "    # Use y_pred for predicted labels, the column represent the predicted class.\n",
        "      conf_mat[val][y_pred[i]] += 1\n",
        "\n",
        "\n",
        "\n",
        "    c_mat = [[ 8,  5,  20],\n",
        "            [ 2, 10,  10],\n",
        "            [ 5,  5, 270]]\n",
        "\n",
        "\n",
        "    c_mat = np.array(conf_mat)\n",
        "\n",
        "    # get the statistics for each class.\n",
        "    tp = TP(c_mat)\n",
        "    fp = FP(c_mat)\n",
        "    tn = TN(c_mat)\n",
        "    fn = FN(c_mat)\n",
        "    acc = accuracy(tp, tn, fp, fn)\n",
        "    acc = [round(x, 3) for x in acc]\n",
        "\n",
        "    macro_acc = macro_ACCURACY(acc)\n",
        "    macro_accuracies.append(macro_acc)\n",
        "\n",
        "\n",
        "    # ----* calcualte micro accuracy *---- #.\n",
        "    micro_c_mat = micro_conf_mat(tp, tn, fp, fn)\n",
        "\n",
        "    for i in range(2):\n",
        "      for j in range(2):\n",
        "        micro_c_mat[i][j] = round(micro_c_mat[i][j], 3)\n",
        "\n",
        "    micro_acc = micro_ACCURACY(micro_c_mat)\n",
        "    micro_accuracies.append(micro_acc)\n",
        "\n",
        "\n",
        "     # ----* calcualte overall accuracy *---- #.\n",
        "    total_cnt = 0\n",
        "    total_correct = 0\n",
        "    for i in range(NUM_CLASSES):\n",
        "      for j in range(NUM_CLASSES):\n",
        "        total_cnt += c_mat[i][j]\n",
        "        if i==j:\n",
        "          total_correct += c_mat[i][j]\n",
        "\n",
        "    overall_acc = total_correct/total_cnt\n",
        "    overall_accuracies.append(overall_acc)\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(depth_values, overall_accuracies, marker='o', label='Overall Accuracy')\n",
        "plt.plot(depth_values, micro_accuracies, marker='x', label='Micro Accuracy')\n",
        "plt.plot(depth_values, macro_accuracies, marker='.', label='Macro Accuracy')\n",
        "\n",
        "plt.xlabel('Max Depth')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('Random Forest Accuracy with varying Max Depth')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "3HsRSgYIEEQi",
        "outputId": "b4d3de09-39ec-4105-9a85-0d9bfad32a09"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1cAAAIjCAYAAADvBuGTAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACFo0lEQVR4nOzdd3gU1R7G8Xd3SYeEFggltID0Ju2CFAsQAZHelY4NLGBBFCk2FBURLzakKR1BxAtKB5UuTRBEeu8ttCSb7Nw/QpYsSUibbBb4fp4nT3bPnJn97cmw7LtnZtZiGIYhAAAAAECGWLO6AAAAAAC4GxCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AeIzu3burWLFiWV0GkOkmTZoki8WigwcPprrvn3/+mfmFucmDDz6oBx98MKvLQBqtXLlSFotFP/zwQ1aXAngswhVwD4p/sxb/ky1bNhUqVEjdu3fXsWPHsro8j3HrOCX8ef3117O6vCS9//77mjdvXprX27VrlywWi3x9fXXx4kXT60LKvvjiC02aNCmry7jrFStWTBaLRQ0bNkxy+bhx45z/zt0daOPDS/yPj4+P8ufPrwcffFDvv/++zpw545Y6pk2bptGjR7vlsYC7TbasLgBA1nn77bdVvHhxRUZGat26dZo0aZL++OMP7dixQ76+vlldnseIH6eEKlSokEXV3N7777+vtm3bqmXLlmlab8qUKQoJCdGFCxf0ww8/qHfv3plTICRJTz75pDp27CgfHx9n2xdffKG8efOqe/fuWVeYmyxevDhLH9/X11crVqzQyZMnFRIS4rJs6tSp8vX1VWRkZBZVJ73wwguqUaOGYmNjdebMGa1Zs0ZDhw7VqFGjNGvWLD388MOZ+vjTpk3Tjh079NJLL2Xq4wB3I8IVcA9r0qSJqlevLknq3bu38ubNqw8//FDz589X+/bts7g6z5FwnMx09epVBQQEmL7dtDIMQ9OmTVPnzp114MABTZ061WPDlaeMWUbZbDbZbLasLsM0af27eHt7Z2I1KXvggQe0ceNGzZw5Uy+++KKz/ejRo/r999/VqlUrzZkzJ8vqq1evntq2bevStm3bNjVu3Fht2rTRzp07VaBAgSyqDsDtcFggAKd69epJkvbt2+dsi46O1pAhQ1StWjUFBQUpICBA9erV04oVK1zWPXjwoCwWiz7++GN98803CgsLk4+Pj2rUqKGNGzcmeqx58+apQoUK8vX1VYUKFfTjjz8mWdPVq1f18ssvKzQ0VD4+PipdurQ+/vhjGYbh0s9isahfv36aPXu2ypUrJz8/P9WuXVvbt2+XJH399dcqWbKkfH199eCDD6bqXJfUWr58uerVq6eAgADlzJlTLVq00K5du1z6DBs2TBaLRTt37lTnzp2VK1cu1a1b17l8ypQpqlatmvz8/JQ7d2517NhRR44ccdnGnj171KZNG4WEhMjX11eFCxdWx44ddenSJecYXL16VZMnT3YeVpSaWZDVq1fr4MGD6tixozp27KjffvtNR48eTdTP4XDos88+U8WKFeXr66vg4GA9+uijiQ6dmjJlimrWrCl/f3/lypVL9evXd5mpsFgsGjZsWKLtFytWzKXe+MMyV61apeeee0758uVT4cKFJUmHDh3Sc889p9KlS8vPz0958uRRu3btkvy7Xrx4Uf3791exYsXk4+OjwoULq2vXrjp79qyuXLmigIAAlzfY8Y4ePSqbzaYRI0YkO3b333+/Wrdu7dJWsWJFWSwW/fXXX862mTNnymKxOPeLW8+5KlasmP7++2+tWrXK+be79ZykqKgoDRgwQMHBwQoICFCrVq1SPEzs448/lsVi0aFDhxItGzRokLy9vXXhwgVJ0u+//6527dqpSJEi8vHxUWhoqPr376/r16+7rNe9e3dlz55d+/btU9OmTZUjRw516dJFQ4cOlZeXV5I1PfXUU8qZM6dzNujWc67iD4ebNWuW3nvvPRUuXFi+vr565JFHtHfv3kTbGzt2rEqUKCE/Pz/VrFlTv//+e5rO4/L19VXr1q01bdo0l/bp06crV65cCg8PT7TOX3/9pe7du6tEiRLy9fVVSEiIevbsqXPnzjn7XL9+XWXKlFGZMmVcxu38+fMqUKCA6tSpo9jY2FTVeKvKlStr9OjRunjxov773/+6LDt27Jh69uyp/Pnzy8fHR+XLl9eECRNc+sSP8cyZM/XGG28oJCREAQEBevzxx11eax588EEtWLBAhw4dcu6Lt54L63A4UvV3Au5FzFwBcIp/o5crVy5nW0REhL799lt16tRJffr00eXLlzV+/HiFh4drw4YNqlKliss2pk2bpsuXL+vpp5+WxWLRyJEj1bp1a+3fv19eXl6S4g4JatOmjcqVK6cRI0bo3Llz6tGjh/ONczzDMPT4449rxYoV6tWrl6pUqaJFixbp1Vdf1bFjx/Tpp5+69P/99981f/589e3bV5I0YsQIPfbYY3rttdf0xRdf6LnnntOFCxc0cuRI9ezZU8uXL0/VuFy6dElnz551acubN68kaenSpWrSpIlKlCihYcOG6fr16/r888/1wAMPaPPmzYnelLRr106lSpXS+++/7wyI7733nt566y21b99evXv31pkzZ/T555+rfv362rJli3LmzKno6GiFh4crKipKzz//vEJCQnTs2DH973//08WLFxUUFKTvv/9evXv3Vs2aNfXUU09JksLCwlJ8flOnTlVYWJhq1KihChUqyN/fX9OnT9err77q0q9Xr16aNGmSmjRpot69eysmJka///671q1b55zZGz58uIYNG6Y6dero7bfflre3t9avX6/ly5ercePGqRrvWz333HMKDg7WkCFDdPXqVUnSxo0btWbNGnXs2FGFCxfWwYMH9eWXX+rBBx/Uzp075e/vL0m6cuWK6tWrp127dqlnz566//77dfbsWc2fP19Hjx5VlSpV1KpVK82cOVOjRo1ymU2aPn26DMNQly5dkq2tXr16mj59uvP++fPn9ffff8tqter3339XpUqVJMXtm8HBwSpbtmyS2xk9erSef/55Zc+eXW+++aYkKX/+/C59nn/+eeXKlUtDhw7VwYMHNXr0aPXr108zZ85Mtr727dvrtdde06xZsxL9PWfNmqXGjRs7/73Pnj1b165d07PPPqs8efJow4YN+vzzz3X06FHNnj3bZd2YmBiFh4erbt26+vjjj+Xv76/atWvr7bff1syZM9WvXz9n3+joaP3www9q06ZNiocbf/DBB7JarXrllVd06dIljRw5Ul26dNH69eudfb788kv169dP9erVU//+/XXw4EG1bNlSuXLlSvQacjudO3dW48aNtW/fPue/k2nTpqlt27bO16qElixZov3796tHjx4KCQnR33//rW+++UZ///231q1bJ4vFIj8/P02ePFkPPPCA3nzzTY0aNUqS1LdvX126dEmTJk3K0Ixl27Zt1atXLy1evFjvvfeeJOnUqVP6z3/+4/yAKTg4WL/88ot69eqliIiIRIf2vffee7JYLBo4cKBOnz6t0aNHq2HDhtq6dav8/Pz05ptv6tKlSzp69KjzNTZ79uwu20jN3wm4ZxkA7jkTJ040JBlLly41zpw5Yxw5csT44YcfjODgYMPHx8c4cuSIs29MTIwRFRXlsv6FCxeM/PnzGz179nS2HThwwJBk5MmTxzh//ryz/aeffjIkGT///LOzrUqVKkaBAgWMixcvOtsWL15sSDKKFi3qbJs3b54hyXj33XddHr9t27aGxWIx9u7d62yTZPj4+BgHDhxwtn399deGJCMkJMSIiIhwtg8aNMiQ5NL3duOU1E/C55IvXz7j3LlzzrZt27YZVqvV6Nq1q7Nt6NChhiSjU6dOLo9x8OBBw2azGe+9955L+/bt241s2bI527ds2WJIMmbPnn3bmgMCAoxu3brdtk9C0dHRRp48eYw333zT2da5c2ejcuXKLv2WL19uSDJeeOGFRNtwOByGYRjGnj17DKvVarRq1cqIjY1Nso9hxP2thg4dmmg7RYsWdak9fvzr1q1rxMTEuPS9du1aovXXrl1rSDK+++47Z9uQIUMMScbcuXOTrXvRokWGJOOXX35xWV6pUiWjQYMGidZLaPbs2YYkY+fOnYZhGMb8+fMNHx8f4/HHHzc6dOjgsq1WrVolem4J98Hy5csn+XjxfRs2bOgyjv379zdsNpvLv6Ok1K5d26hWrZpL24YNGxKNVVJjOmLECMNisRiHDh1ytnXr1s2QZLz++utJPlatWrVc2ubOnWtIMlasWOFsa9CggctzXbFihSHJKFu2rMvrzWeffWZIMrZv324YhmFERUUZefLkMWrUqGHY7XZnv0mTJhmSUvx7GUbcftasWTMjJibGCAkJMd555x3DMAxj586dhiRj1apVzjHfuHHjbcdn+vTphiTjt99+c2kfNGiQYbVajd9++825j4wePTrF2uLH4Xb/zitXrmzkypXLeb9Xr15GgQIFjLNnz7r069ixoxEUFOSsO37bhQoVcnk9nDVrliHJ+Oyzz5xtzZo1c3ktvrW+lP5OwL2MwwKBe1jDhg0VHBys0NBQtW3bVgEBAZo/f77Lp782m815foTD4dD58+cVExOj6tWra/PmzYm22aFDB5eZr/hDDffv3y9JOnHihLZu3apu3bopKCjI2a9Ro0YqV66cy7YWLlwom82mF154waX95ZdflmEY+uWXX1zaH3nkEZeZolq1akmS2rRpoxw5ciRqj68pJWPHjtWSJUtcfhI+l+7duyt37tzO/pUqVVKjRo20cOHCRNt65plnXO7PnTtXDodD7du319mzZ50/ISEhKlWqlPPwy/ixWrRoka5du5aqulPjl19+0blz59SpUydnW6dOnbRt2zb9/fffzrY5c+bIYrFo6NChibZhsVgkxR3q6XA4NGTIEFmt1iT7pEefPn0Sfdrv5+fnvG2323Xu3DmVLFlSOXPmdNkv58yZo8qVK6tVq1bJ1t2wYUMVLFhQU6dOdS7bsWOH/vrrLz3xxBO3rS1+//7tt98kxc1Q1ahRQ40aNdLvv/8uKe6wxB07djj7ptdTTz3lMo716tVTbGxskof8JdShQwdt2rTJ5XDfmTNnysfHRy1atHC2JRzTq1ev6uzZs6pTp44Mw9CWLVsSbffZZ59N1Na1a1etX7/e5bGmTp2q0NBQNWjQIMXn2KNHD5fzsW59/fjzzz917tw59enTR9my3Tz4pkuXLi6vO6lhs9nUvn1758xjfJ3J/Z0Sjk9kZKTOnj2r//znP5KU6LVw2LBhKl++vLp166bnnntODRo0SPQ6ll7Zs2fX5cuXJcXN7s+ZM0fNmzeXYRguryHh4eG6dOlSotq6du3q8nrYtm1bFShQIMnXq+Sk9HcC7mWEK+AeFh8afvjhBzVt2lRnz551uXpZvMmTJ6tSpUry9fVVnjx5FBwcrAULFjjP9UmoSJEiLvfj3/DEn9cR/0awVKlSidYtXbq0y/1Dhw6pYMGCLm8EJDkPrbr1TeWtjx0fSEJDQ5Nsj68pJTVr1lTDhg1dfhI+/q11x9d49uxZ52Fs8W696uCePXtkGIZKlSql4OBgl59du3bp9OnTzvUGDBigb7/9Vnnz5lV4eLjGjh2b5N8gLaZMmaLixYvLx8dHe/fu1d69exUWFiZ/f3+XsLFv3z4VLFjQJUTeat++fbJarYlCckbdOmZS3LktQ4YMcZ6LlzdvXgUHB+vixYsuY7Jv374Ur+xotVrVpUsXzZs3zxlc468Y165du9uumz9/fpUqVcoZpH7//XfVq1dP9evX1/Hjx7V//36tXr1aDocjw+EqpX9byWnXrp2sVqvz8EHDMDR79mw1adJEgYGBzn6HDx92flCQPXt2BQcHOwPRrftZtmzZkjwEr0OHDvLx8XHuO5cuXdL//vc/denSJVUBO7WvHyVLlkxUT3q+I69z587auXOntm3bpmnTpqljx47J1nn+/Hm9+OKLyp8/v/z8/BQcHOzcN28dH29vb02YMEEHDhzQ5cuXNXHixAx9wJDQlStXnK+JZ86c0cWLF/XNN98kev3o0aOHJDlfQ+Ld+tprsVhUsmTJNJ2Hmt59EbgXcM4VcA+rWbOm81yZli1bqm7duurcubN2797tPMZ+ypQp6t69u1q2bKlXX31V+fLlc57kn/DT6XjJnU9g3HIBisyQ3GNnZU23SvjptxQ3G2ixWPTLL78kWWfCcx0++eQTde/eXT/99JMWL16sF154QSNGjNC6devSdK5JvIiICP3888+KjIxMMuxOmzbNeX6GOyR3ov+tYybFnX80ceJEvfTSS6pdu7aCgoJksVjUsWNHORyOND92165d9dFHH2nevHnq1KmTpk2bpscee8xldjU5devW1bJly3T9+nVt2rRJQ4YMUYUKFZQzZ079/vvv2rVrl7Jnz66qVaumua6E0rsfFyxYUPXq1dOsWbP0xhtvaN26dTp8+LA+/PBDZ5/Y2Fg1atRI58+f18CBA1WmTBkFBATo2LFj6t69e6Ix9fHxSTQ7KcW9yX7sscc0depUDRkyRD/88IOioqJSnAHM6HNMr1q1aiksLEwvvfSSDhw4oM6dOyfbt3379lqzZo1effVVValSRdmzZ5fD4dCjjz6a5D63aNEiSXGzXHv27EnyQ4K0stvt+vfff50fGMQ/7hNPPKFu3boluU78eX9m8qTXVMDTEK4ASJIzMD300EP673//6/yS3B9++EElSpTQ3LlzXd5kJ3V4WGoULVpUUtyMza12796dqO/SpUt1+fJll9mrf/75x2VbWSX+8W+tW4qrMW/evClenjosLEyGYah48eK67777UnzMihUrqmLFiho8eLDWrFmjBx54QF999ZXeffddSWk7/G7u3LmKjIzUl19+6bxAR7zdu3dr8ODBWr16terWrauwsDAtWrRI58+fT3b2KiwsTA6HQzt37kx0oZOEcuXKleiLiqOjo3XixIlU1/7DDz+oW7du+uSTT5xtkZGRibYbFhamHTt2pLi9ChUqqGrVqpo6daoKFy6sw4cP6/PPP09VLfXq1dPEiRM1Y8YMxcbGqk6dOrJarapbt64zXNWpUyfFCxlkZojt0KGDnnvuOe3evVszZ86Uv7+/mjdv7ly+fft2/fvvv5o8ebK6du3qbI8/BDYtunbtqhYtWmjjxo2aOnWqqlatqvLly5vyPOL/ze3du1cPPfSQsz0mJkYHDx5MV5Do1KmT3n33XZUtWzbZ/fbChQtatmyZhg8friFDhjjbk3odk+KuLPj222+rR48e2rp1q3r37q3t27enKqzfzg8//KDr1687r2YYHBysHDlyKDY2NtkvRb7VrTUbhqG9e/e6jJ27PlAB7kYcFgjA6cEHH1TNmjU1evRo5yWT498QJvxEcv369Vq7dm26HqNAgQKqUqWKJk+e7HIozZIlS7Rz506Xvk2bNlVsbGyiyw5/+umnslgsatKkSbpqMEvC55LwTf2OHTu0ePFiNW3aNMVttG7dWjabTcOHD0/0qa9hGM7LPEdERCgmJsZlecWKFWW1WhUVFeVsCwgISBQwkjNlyhSVKFFCzzzzjNq2bevy88orryh79uzOw7vatGkjwzA0fPjwRNuJr7tly5ayWq16++23E32Sn/C5hYWFOc9RivfNN9+k6RLVNpst0Xh9/vnnibbRpk0bbdu2LclL/d+6/pNPPqnFixdr9OjRypMnT6r3r/jD/T788ENVqlTJ+Qa6Xr16WrZsmf78889UHRKYlr9dWrVp00Y2m03Tp0/X7Nmz9dhjj7kE/6T+nRuGoc8++yzNj9WkSRPnd+atWrUq1bNWqVG9enXlyZNH48aNc/n3MHXq1HQfkta7d28NHTrUJajfKqnxkeKu8ngru92u7t27q2DBgvrss880adIknTp1Sv37909XffG2bduml156Sbly5XJeEdVms6lNmzaaM2dOkh8iJHVZ/O+++855zpYUF9hOnDjhsr8HBARk+JBj4F7FzBUAF6+++qratWunSZMm6ZlnntFjjz2muXPnqlWrVmrWrJkOHDigr776SuXKldOVK1fS9RgjRoxQs2bNVLduXfXs2VPnz5/X559/rvLly7tss3nz5nrooYf05ptv6uDBg6pcubIWL16sn376SS+99FKqLjOe2T766CM1adJEtWvXVq9evZyXYg8KCkryu5xuFRYWpnfffVeDBg1yXlI6R44cOnDggH788Uc99dRTeuWVV7R8+XL169dP7dq103333aeYmBh9//33zjdX8apVq6alS5dq1KhRKliwoIoXL+68gEdCx48f14oVK5I9yd7Hx0fh4eGaPXu2xowZo4ceekhPPvmkxowZoz179jgPhfr999/10EMPqV+/fipZsqTefPNNvfPOO6pXr55at24tHx8fbdy4UQULFnR+X1Tv3r31zDPPqE2bNmrUqJG2bdumRYsWJZo9u53HHntM33//vYKCglSuXDmtXbtWS5cuVZ48eVz6vfrqq/rhhx/Url079ezZU9WqVdP58+c1f/58ffXVV6pcubKzb+fOnfXaa6/pxx9/1LPPPpvk5biTUrJkSYWEhGj37t16/vnnne3169fXwIEDJSlV4apatWr68ssv9e6776pkyZLKly+fHn744VTVkJJ8+fLpoYce0qhRo3T58mV16NDBZXmZMmUUFhamV155RceOHVNgYKDmzJmTrsDi5eWljh076r///a9sNpvLxVIyytvbW8OGDdPzzz+vhx9+WO3bt9fBgwc1adIkhYWFpWvGpWjRoin+Ww0MDFT9+vU1cuRI2e12FSpUSIsXL9aBAwcS9X333Xe1detWLVu2TDly5FClSpU0ZMgQDR48WG3btk3Vhy6///67IiMjFRsbq3Pnzmn16tWaP3++goKC9OOPPyokJMTZ94MPPtCKFStUq1Yt9enTR+XKldP58+e1efNmLV26VOfPn3fZdu7cuVW3bl316NFDp06d0ujRo1WyZEn16dPH2adatWqaOXOmBgwYoBo1aih79uwuM50AbsN9FyYE4CmSusxwvNjYWCMsLMwICwszYmJiDIfDYbz//vtG0aJFDR8fH6Nq1arG//73P6Nbt24ul+qNvxT7Rx99lGibSuLS23PmzDHKli1r+Pj4GOXKlTPmzp2baJuGYRiXL182+vfvbxQsWNDw8vIySpUqZXz00Ucul6SOf4y+ffu6tCVXU2oud5zSOCW0dOlS44EHHjD8/PyMwMBAo3nz5s5Lc8eLvxT7mTNnktzGnDlzjLp16xoBAQFGQECAUaZMGaNv377G7t27DcMwjP379xs9e/Y0wsLCDF9fXyN37tzGQw89ZCxdutRlO//8849Rv359w8/Pz5CU7GXZP/nkE0OSsWzZsmSfV/zlrX/66SfDMOIuy//RRx8ZZcqUMby9vY3g4GCjSZMmxqZNm1zWmzBhglG1alXDx8fHyJUrl9GgQQNjyZIlzuWxsbHGwIEDjbx58xr+/v5GeHi4sXfv3mQvxZ7U+F+4cMHo0aOHkTdvXiN79uxG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TJkyzj6tWrVSjRo1TC0UAAAAdxfDMHTdHpuqvrEOQ0Pn/62kjo0yJFkkDZu/Uw+UzCubNeUrv/l52VJ1hbjz589r8eLFGjx4sPP7kOKFhISoS5cumjlzpr744gt16dJFI0aM0L59+xQWFiZJ+vvvv/XXX39pzpw5kqSpU6dqyJAh+u9//6uqVatqy5Yt6tOnjwICAtStWzfntl9//XV98sknqlq1qnx9fRUZGalq1app4MCBCgwM1IIFC/Tkk08qLCxMNWvWTPF5JGfFihW6du2aGjZsqEKFCqlOnTr69NNPFRAQICkuDD3yyCPq2bOnPvvsM2XLlk0rVqxQbGzc323QoEEaN26cPv30U9WtW1cnTpzQP//8k6Yarl27pg8//FDffvut8uTJo3z58mnv3r3q2LGjxo4dK4vFok8++URNmzbVnj17lCNHDjkcDjVp0kSXL1/WlClTFBYWpp07d8pmsykgIEAdO3bUxIkT1bZtW+fjxN/PkSNHuscrJWkOVzVq1FCjRo305ZdfqmXLlvLy8krUp3jx4urYsaMpBQIAAODudN0eq3JDFpmyLUPSyYhIVRy2OFX9d74dLn/vlN8K79mzR4ZhqHTp0kkuL1u2rC5cuKAzZ86ofPnyqly5sqZNm6a33npLUlyYqlWrlkqWLClJGjp0qD755BO1bt1aUtz75p07d+rrr792CVcvvfSSs0+8V155xXn7+eef16JFizRr1qwMhavx48erY8eOstlsqlChgkqUKKHZs2ere/fukqSRI0eqevXqLrNr5cuXlyRdvnxZn332mf773/86aw8LC1PdunXTVIPdbtcXX3yhypUrO9sefvhhVa9eXYGBgbJarfrmm2+UM2dOrVq1So899piWLl2qDRs2aNeuXbrvvvskSSVKlHCu37t3b9WpU0cnTpxQgQIFdPr0aS1cuFBLly5N1zilVpoPC9y/f79+/fVXtWvXLslgJUkBAQGaOHFihosDAAAAPEFqryfQpUsXTZs2zbnO9OnTnafLXL16Vfv27VOvXr2UPXt258+7776rffv2uWynevXqLvdjY2P1zjvvqGLFisqdO7eyZ8+uRYsWZejrkS5evKi5c+fqiSeecLY98cQTLocGxs9cJWXXrl2KiopKdnlqeXt7q1KlSi5tp06d0osvvqjSpUsrKChIgYGBunLlivP5bt26VYULF3YGq1vVrFlT5cuX1+TJkyVJU6ZMUdGiRVW/fv0M1ZqSNM9cnT59WidPnkx0POn69etls9kS7QgAAABAUvy8bNr5dniq+m44cF7dJ25Msd+kHjVUs3juVD12apQsWVIWi0W7d+9OcvmuXbuUK1cuBQcHS5I6deqkgQMHavPmzbp+/bqOHDmiDh06SJKuXLkiSRo3blyi99I2m2s98Yflxfvoo4/02WefafTo0apYsaICAgL00ksvKTo6OlXPIynTpk1TZGSkSy2GYcjhcOjff//Vfffdl+hQyIRut0ySrFarc5vxkro4h5+fX6JDNLt3764zZ87o008/VfHixeXj46PatWs7n29Kjy3FzV6NHTtWr7/+uiZOnKgePXpk+pdYp3nmqm/fvjpy5Eii9mPHjqlv376mFAUAAIC7n8Vikb93tlT91CsVrAJBvkrurbFFUoEgX9UrFZyq7aX2TXaePHnUsGFDTZgwQdevX3dZdvLkSU2dOlUdOnRwbq9w4cJq0KCBpk6dqqlTp6pRo0bKly+fJCl//vwqWLCg9u/fr5IlS7r8FC9e/LZ1rF69Wi1atNATTzyhypUrq0SJEvr3339T9RySM378eL388svaunWr82fbtm2qV6+eJkyYIEmqVKmSli1bluT6pUqVkp+fX7LL4wPniRM3LzSydevWVNW2Zs0aPfXUU2ratKnKly8vHx8fnT171rm8UqVKOnr06G3H4IknntChQ4c0ZswY7dy50+Wwy8yS5nC1c+dO3X///Ynaq1atqp07d5pSFAAAAJCQzWrR0OblJClRwIq/P7R5uVRdzCKtPv/8c0VFRalJkyb67bffdOTIEf36669q1KiRChUq5Lw6XrwuXbpoxowZmj17dqIraA8fPlwjRozQmDFj9O+//2r79u2aOHGiRo0addsaSpUqpSVLlmjNmjXatWuXnn76aZ06dSrdz2nr1q3avHmzevfurQoVKrj8dOrUSZMnT1ZMTIwGDRqkjRs36rnnntNff/2lf/75R19++aXOnj0rX19fDRw4UK+99pq+++477du3T+vWrXMeVliyZEmFhoZq2LBh2rNnjxYsWKBPPvkkVfWVKlVKs2bN0q5du7R+/Xp16dLFZbaqQYMGql+/vtq0aaMlS5bowIED+uWXX/Trr786++TKlUutW7fWq6++qsaNG6tw4cLpHq/USnO4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/qtZs2ZpyZIlatiwoSQleYXtt99+W40aNZIkXb16VV9++aUmTZqkJk2aSJLGjRunJUuWaPz48Xr11VeTfCyHw6FJkybp888/lyR17NhRL7/8sg4cOKDixYtLkt599129/PLLevHFF53r1ahRQ5K0dOlSbdiwQbt27dJ9992XbK0pCQgI0Lfffitvb29nW8IQVKJECY0ZM0Y1atTQlStXlD17do0dO1ZBQUGaMWOGvLy85HA4FBISosDAQElSr169NHHiROdz//nnnxUZGan27dunub7MlOZw1bJly0woAwAAAPBswcHBatasmSZNmiTDMNSsWTPlzZv3tuts3bpVNptNDRo0uG2/6tWrO2/v27dPdrtdDzzwgLPNy8tLNWvW1K5du5LdxpIlS3T16lU1bdpUUtx30jZq1EgTJkzQO++8o9OnT+v48eN65JFHkq21cOHCzmCVXhUrVnQJVpK0adMmDRs2TNu2bdOFCxecVx8/fPiwypUrp61bt6pevXry8vJKcpvdu3fX4MGDtW7dOv3nP//RpEmT1L59+0QTPVktzeFq6NChmVEHAAAA7jVe/nEzSGn1x6dxs1Q2byk2Ou6QwLr90/7Y6dCzZ0/nOUpjx45Nsb+fn1+qtmtGSBg/frzOnz/v8pgOh0N//fWXhg8fnmItKS23Wq0ybjmc0m63J+p363O5evWqwsPDFR4erqlTpyo4OFiHDx9WeHi484IXKT12vnz51Lx5c02cOFHFixfXL7/8ctsZw6yS5nOuAAAAAFNYLHGH5qXlZ+3YuGD10JvSW2fifv/2UVx7WraTzitdP/roo4qOjpbdbld4eHiK/StWrCiHw6FVq1al+jHCwsLk7e2t1atXO9vsdrs2btyocuXKJbnOuXPn9NNPP2nGjBnaunWr82fLli26cOGCFi9erBw5cqhYsWJatmxZktuoVKmSjh49qn///TfJ5cHBwTp58qRLwNq6dWuKz+eff/7RuXPn9MEHH6hevXoqU6ZMootZVKpUSb///nuSYS1e7969NXPmTH3zzTcKCwtzmdnzFGkOV1arVTabLdkfAAAAIFPEXxXwoTelBq/FtTV4Le5+UlcRzAQ2m027du3Szp07U/Xet1ixYurWrZt69uypefPm6cCBA1q5cqVmzZqV7DoBAQF69tln9eqrr+rXX3/Vzp071adPH127dk29evVKcp3vv/9eefLkUfv27VWhQgXnT+XKldW0aVPnhS2GDRumTz75RGPGjNGePXu0efNm5zlaDRo0UP369dWmTRstWbJEBw4c0C+//KJff/1VUtxVEM+cOaORI0dq3759Gjt2rH755ZcUx6BIkSLy9vbW559/rv3792v+/PmJvgesX79+ioiIUMeOHfXnn39qz549mjFjhnbv3u3sEx4ersDAQL377rupupBIVkhzuPrxxx81d+5c58/MmTP1+uuvq0CBAvrmm28yo0YAAABAcsS6Bqt48QHL4Z6vCwoMDHReaCE1vvzyS7Vt21bPPfecypQpoz59+ujq1au3XeeDDz5QmzZt9OSTT+r+++/X3r17tWjRIuXKlSvJ/hMmTFCrVq2S/O7ZNm3aaP78+Tp79qy6deum0aNH64svvlD58uX12GOPac+ePc6+c+bMUY0aNdSpUyeVK1dOr732mvNrmMqWLasvvvhCY8eOVeXKlbVhwwa98sorKT7/4OBgTZo0SbNnz1a5cuX0wQcf6OOPP3bpkydPHi1fvlxXrlxRgwYNVKNGDX333Xcu52BZrVZ1795dsbGx6tq1a4qPmxUsxq0HTqbTtGnTNHPmTP30009mbC5LRUREKCgoSJcuXUrTPxyknd1u18KFC9W0adNkT2CEuRhz92K83Y8xdz/G3L3u1PGOjIx0XrXO19c3q8tJE4fDoYiICAUGBspq5awad0huzHv16qUzZ85o/vz5pj7e7fbPtGSDNF/QIjn/+c9/9NRTT5m1OQAAAACQJF26dEnbt2/XtGnTTA9WZjIlXF2/fl1jxoxRoUKFzNgcAAAAADi1aNFCGzZs0DPPPOP8PjBPlOZwlStXLpdjOQ3D0OXLl+Xv768pU6aYWhwAAAAAeOJl15OS5nD16aefuoQrq9Wq4OBg1apVK9kT7AAAAADgbpfmcNW9e/dMKAMAAAAA7mxpvtzJxIkTNXv27ETts2fP1uTJk00pCgAAAADuNGkOVyNGjFDevHkTtefLl0/vv/++KUUBAAAAwJ0mzeHq8OHDKl68eKL2okWL6vDhw6YUBQAAAAB3mjSHq3z58umvv/5K1L5t2zblyZPHlKIAAAAA4E6T5nDVqVMnvfDCC1qxYoViY2MVGxur5cuX68UXX1THjh0zo0YAAAAA8HhpDlfvvPOOatWqpUceeUR+fn7y8/NT48aN9fDDD3POFQAAAO46PXr0UK5cufTss88mWta3b19ZLBaPv6L2008/LZvNluSF6WCeNIcrb29vzZw5U7t379bUqVM1d+5c7du3TxMmTJC3t3dm1AgAAABkqUKFCmnmzJm6fv26sy0yMlLTpk1TkSJFMvWxo6OjM7T+tWvXNGPGDL322muaMGGCSVWlX0afjydLc7iKV6pUKbVr106PPfaYihYtamZNAAAAwG2dvHpSG05s0MmrJ93yeJUrV1ZoaKjmzp3rbJs7d66KFCmiqlWruvT99ddfVbduXeXMmVN58uTRY489pn379rn0OXr0qDp16qTcuXMrICBA1atX1/r16yVJw4YNU5UqVfTtt9+qePHi8vX1lRR3YbkWLVooe/bsCgwMVPv27XXq1KkUa589e7bKlSun119/Xb/99puOHDnisjwqKkoDBw5UaGiofHx8VLJkSY0fP965/O+//9Zjjz2mwMBA5ciRQ/Xq1XM+nwcffFAvvfSSy/ZatmzpMpNXrFgxvfPOO+ratasCAwP11FNPSZIGDhyo++67T/7+/ipRooTeeust2e12l239/PPPqlGjhnx9fZU3b161atVKkvT222+rQoUKiZ5rlSpV9NZbb6U4JpklzeGqTZs2+vDDDxO1jxw5Uu3atTOlKAAAANz9DMPQNfu1NP/M+GeGwn8IV6/FvRT+Q7hm/DMjzdswDCPN9fbo0UMTJ0503p8wYYJ69OiRqN/Vq1c1YMAA/fnnn1q2bJmsVqtatWolh8MhSbpy5YoaNGigY8eOaf78+dq2bZtee+0153JJ2rt3r+bMmaO5c+dq69atcjgcatGihc6fP69Vq1ZpyZIl2r9/vzp06JBi3ePHj9cTTzyhoKAgNWnSRJMmTXJZ3rVrV02fPl1jxozRrl279PXXXyt79uySpGPHjql+/fry8fHR8uXLtWnTJvXs2VMxMTFpGruPP/5YlStX1pYtW5zhJ0eOHJo0aZJ27typzz77TOPGjdOnn37qXGfRokVq06aNmjZtqi1btmjZsmWqWbOmJKlnz57atWuXNm7c6Oy/ZcsW/fXXX0n+TdwlW1pX+O233zRs2LBE7U2aNNEnn3xiRk0AAAC4B1yPua5a02plaBsOOfTe+vf03vr30rTe+s7r5e/ln6Z1unTpojfeeEOHDh2SJK1evVozZszQypUrXfq1adPG5f6ECRMUHBysnTt3qkKFCpo2bZrOnDmjjRs3Knfu3JKkkiVLuqwTHR2t7777TsHBwZKkJUuWaPv27Tpw4IBCQ0MlSd99953Kly+vjRs3qkaNGknWvGfPHq1bt8454/bEE09owIABGjx4sCwWi/7991/NmjVLS5YsUcOGDSVJJUqUcK4/duxYBQUFacaMGfLy8pIk3XfffWkaN0l6+OGH9fLLL7u0DR482Hm7WLFieuWVV5yHL0rSJ598og4dOmj48OHOfpUrV5YkFS5cWOHh4Zo4caLzuU+cOFENGjRwqd/d0jxzdeXKlSTPrfLy8lJERIQpRQEAAACeJjg4WM2aNdOkSZM0ceJENWvWTHnz5k3Ub8+ePerUqZNKlCihwMBAFStWTJKc3wm7detWVa1a1RmsklK0aFFnsJKkXbt2KTQ01BmsJKlcuXLKmTOndu3alex2JkyYoPDwcGedTZs21aVLl7R8+XJnLTabTQ0aNEhy/a1bt6pevXrOYJVe1atXT9Q2c+ZMPfDAAwoJCVH27Nk1ePBgl+/N3bFjhx5++OFkt9mnTx9Nnz5dkZGRio6O1rRp09SzZ88M1ZlRaZ65qlixombOnKkhQ4a4tM+YMUPlypUzrTAAAADc3fyy+Wl95/VpWufUtVNqOa+lHLp5CJ3VYtW8FvOU3z9/mh47PXr27Kl+/fpJipvVSUrz5s1VtGhRjRs3TgULFpTD4VCFChWcF3Lw80v5sQMCAtJVX0KxsbGaPHmyTp48qWzZsrm0T5gwwXn179tJabnVak10iOWt501JiZ/P2rVr1aVLFw0fPlzh4eHO2bGER8LFn2uWnObNm8vHx0c//vijvL29Zbfb1bZt29uuk9nSHK7eeusttW7dWvv27XMmyWXLlmnatGn64YcfTC8QAAAAdyeLxZLmQ/OKBxXX0DpDNXztcDkMh6wWq4bWHqriQcUzqUpXjz76qKKjo2WxWBQeHp5o+blz57R7926NGzdO9erVkyT98ccfLn0qVaqkb7/9VufPn7/t7FVCZcuW1ZEjR3TkyBHn7NXOnTt18eLFZCc4Fi5cqMuXL2vLli2y2WzO9h07dqhHjx66ePGiKlasKIfDoVWrVjkPC7y11smTJ8tutyc5exUcHKwTJ04478fGxmrHjh166KGHbvt81qxZo6JFi+rNN990tsUfbhmvfPnyWr58uXr16pXkNrJly6Zu3bpp4sSJ8vb2VseOHVMVXDNTmsNV8+bNNW/ePL3//vv64Ycf5Ofnp8qVK2v58uWp3jkAAACA9GpdqrXqFKyjI5ePKDRHqEICQtz22DabzXkYXsLAEi9XrlzKkyePvvnmGxUoUECHDx/W66+/7tKnU6dOev/999WyZUuNGDFCBQoU0JYtW1SwYEHVrl07ycdt2LChKlasqC5dumj06NGKiYnRc889pwYNGiR5yJ0UdyGLZs2aOc9TileuXDn1799fU6dOVd++fdWtWzf17NlTY8aMUeXKlXXo0CGdPn1a7du3V79+/fT555+rY8eOGjRokIKCgrRu3TrVrFlTpUuX1sMPP6wBAwZowYIFCgsL06hRo3Tx4sUUx7FUqVI6fPiwZsyYoRo1amjBggX68ccfXfoMHDhQLVq0UMmSJdWxY0fFxMRo4cKFGjhwoLNP7969VbZsWUlx58BltXRdir1Zs2ZavXq1rl69qv3796t9+/Z65ZVXEv3hAAAAgMwQEhCiGiE13Bqs4gUGBiowMDDJZVarVTNmzNCmTZtUoUIF9e/fXx999JFLH29vby1evFj58uVT06ZNVbFiRX3wwQdJhrV4FotFP/30k3LlyqX69eurYcOGKlGihGbOnJlk/1OnTmnBggWJLq4RX2OrVq2cl1v/8ssv1bZtWz333HMqU6aM+vTpo6tXr0qS8uTJo+XLlzuvcFitWjWNGzfOOYvVs2dPdevWTV27dnVeTCKlWStJevzxx9W/f3/169dPVapU0Zo1axJdQr1u3bqaOXOm5s+frypVqujhhx/Whg0bXPqUKlVKderUUZkyZVSrVsYujmIGi5Ge61Aq7qqB48eP15w5c1SwYEG1bt1abdq0SfZKJXeSiIgIBQUF6dKlS8n+w4E57Ha7Fi5cqKZNm2b4REmkDmPuXoy3+zHm7seYu9edOt6RkZE6cOCAy/c23SkcDociIiIUGBgoqzXdXxOLNEjtmBuGoVKlSum5557TgAED0v14t9s/05IN0nRY4MmTJzVp0iSNHz9eERERat++vaKiojRv3jwuZgEAAADAbc6cOaMZM2bo5MmTWfrdVgmlOlw1b95cv/32m5o1a6bRo0fr0Ucflc1m01dffZWZ9QEAAABAIvny5VPevHn1zTffKFeuXFldjqQ0hKtffvlFL7zwgp599lmVKlUqM2sCAAAAgNtK59lNmSrVB43+8ccfunz5sqpVq6ZatWrpv//9r86ePZuZtQEAAADAHSPV4eo///mPxo0bpxMnTujpp5/WjBkznF+KtmTJEl2+fDkz6wQAAMBdwBNnGwCz9ss0X+4kICBAPXv21B9//KHt27fr5Zdf1gcffKB8+fLp8ccfN6UoAAAA3F3ir2x47dq1LK4ESCx+v8zoFTjT/CXCCZUuXVojR47UiBEj9PPPP2vChAkZKgYAAAB3J5vNppw5c+r06dOSJH9/f1ksliyuKnUcDoeio6MVGRnJpdjdxF1jbhiGrl27ptOnTytnzpy3/a6x1MhQuIpns9nUsmVLtWzZ0ozNAQAA4C4UEhL3hb/xAetOYRiGrl+/Lj8/vzsmEN7p3D3mOXPmdO6fGWFKuAIAAABSYrFYVKBAAeXLl092uz2ry0k1u92u3377TfXr17+jvrj5TubOMffy8srwjFU8whUAAADcymazmfZm1h1sNptiYmLk6+tLuHKTO3XMOWgUAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwQZaHq7Fjx6pYsWLy9fVVrVq1tGHDhtv2Hz16tEqXLi0/Pz+Fhoaqf//+ioyMzNA2AQAAACCjsjRczZw5UwMGDNDQoUO1efNmVa5cWeHh4Tp9+nSS/adNm6bXX39dQ4cO1a5duzR+/HjNnDlTb7zxRrq3CQAAAABmyNJwNWrUKPXp00c9evRQuXLl9NVXX8nf318TJkxIsv+aNWv0wAMPqHPnzipWrJgaN26sTp06ucxMpXWbAAAAAGCGbFn1wNHR0dq0aZMGDRrkbLNarWrYsKHWrl2b5Dp16tTRlClTtGHDBtWsWVP79+/XwoUL9eSTT6Z7m5IUFRWlqKgo5/2IiAhJkt1ul91uz9DzxO3Fjy/j7D6MuXsx3u7HmLsfY+5ejLf7Mebu50ljnpYasixcnT17VrGxscqfP79Le/78+fXPP/8kuU7nzp119uxZ1a1bV4ZhKCYmRs8884zzsMD0bFOSRowYoeHDhydqX7x4sfz9/dP61JAOS5YsyeoS7jmMuXsx3u7HmLsfY+5ejLf7Mebu5wljfu3atVT3zbJwlR4rV67U+++/ry+++EK1atXS3r179eKLL+qdd97RW2+9le7tDho0SAMGDHDej4iIUGhoqBo3bqzAwEAzSkcy7Ha7lixZokaNGsnLyyury7knMObuxXi7H2Pufoy5ezHe7seYu58njXn8UW2pkWXhKm/evLLZbDp16pRL+6lTpxQSEpLkOm+99ZaefPJJ9e7dW5JUsWJFXb16VU899ZTefPPNdG1Tknx8fOTj45Oo3cvLK8v/mPcKxtr9GHP3YrzdjzF3P8bcvRhv92PM3c8Txjwtj59lF7Tw9vZWtWrVtGzZMmebw+HQsmXLVLt27STXuXbtmqxW15JtNpskyTCMdG0TAAAAAMyQpYcFDhgwQN26dVP16tVVs2ZNjR49WlevXlWPHj0kSV27dlWhQoU0YsQISVLz5s01atQoVa1a1XlY4FtvvaXmzZs7Q1ZK2wQAAACAzJCl4apDhw46c+aMhgwZopMnT6pKlSr69ddfnRekOHz4sMtM1eDBg2WxWDR48GAdO3ZMwcHBat68ud57771UbxMAAAAAMkOWX9CiX79+6tevX5LLVq5c6XI/W7ZsGjp0qIYOHZrubQIAAABAZsjSLxEGAAAAgLsF4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAEzgEeFq7NixKlasmHx9fVWrVi1t2LAh2b4PPvigLBZLop9mzZo5+3Tv3j3R8kcffdQdTwUAAADAPSpbVhcwc+ZMDRgwQF999ZVq1aql0aNHKzw8XLt371a+fPkS9Z87d66io6Od98+dO6fKlSurXbt2Lv0effRRTZw40Xnfx8cn854EAAAAgHtels9cjRo1Sn369FGPHj1Urlw5ffXVV/L399eECROS7J87d26FhIQ4f5YsWSJ/f/9E4crHx8elX65cudzxdAAAAADco7J05io6OlqbNm3SoEGDnG1Wq1UNGzbU2rVrU7WN8ePHq2PHjgoICHBpX7lypfLly6dcuXLp4Ycf1rvvvqs8efIkuY2oqChFRUU570dEREiS7Ha77HZ7Wp8W0iB+fBln92HM3Yvxdj/G3P0Yc/divN2PMXc/TxrztNRgMQzDyMRabuv48eMqVKiQ1qxZo9q1azvbX3vtNa1atUrr16+/7fobNmxQrVq1tH79etWsWdPZPmPGDPn7+6t48eLat2+f3njjDWXPnl1r166VzWZLtJ1hw4Zp+PDhidqnTZsmf3//DDxDAAAAAHeya9euqXPnzrp06ZICAwNv2zfLz7nKiPHjx6tixYouwUqSOnbs6LxdsWJFVapUSWFhYVq5cqUeeeSRRNsZNGiQBgwY4LwfERGh0NBQNW7cOMUBRMbY7XYtWbJEjRo1kpeXV1aXc09gzN2L8XY/xtz9GHP3YrzdjzF3P08a8/ij2lIjS8NV3rx5ZbPZdOrUKZf2U6dOKSQk5LbrXr16VTNmzNDbb7+d4uOUKFFCefPm1d69e5MMVz4+Pkle8MLLyyvL/5j3Csba/Rhz92K83Y8xdz/G3L0Yb/djzN3PE8Y8LY+fpRe08Pb2VrVq1bRs2TJnm8Ph0LJly1wOE0zK7NmzFRUVpSeeeCLFxzl69KjOnTunAgUKZLhmAAAAAEhKll8tcMCAARo3bpwmT56sXbt26dlnn9XVq1fVo0cPSVLXrl1dLngRb/z48WrZsmWii1RcuXJFr776qtatW6eDBw9q2bJlatGihUqWLKnw8HC3PCcAAAAA954sP+eqQ4cOOnPmjIYMGaKTJ0+qSpUq+vXXX5U/f35J0uHDh2W1umbA3bt3648//tDixYsTbc9ms+mvv/7S5MmTdfHiRRUsWFCNGzfWO++8w3ddAQAAAMg0WR6uJKlfv37q169fkstWrlyZqK106dJK7iKHfn5+WrRokZnlAQAAAECKsvywQAAAAAC4GxCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCsAAAAAMAHhCgAAAABMQLgCAAAAABMQrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAADgUU5dO6X99v06de1UVpeSJoQrAAAAAB5j7p65ajavmSZcnaBm85pp7p65WV1SqmXL6gIAAAAAZJ2TV0/qcMRhFQksopCAkAxtyzAMXY+5riv2K7piv6Kr0VfjftsT/I6+ef/WtktRl3Ty2knn9hxyaPja4apTsE6Ga3MHwhUAAABwj5q7Z66Grx0uh+GQVVa9VO0l1StUL1Eguhx9OdmA5PwdfVVXY67KYThMrdFhOHTk8hHCFQAAAICb0jtLFOuIVWRspK7HXNf1mOuKjIl0+X099rqu2687+zjbb+0fG+nsdyX6SqJZolGbRmnUplEZfp42i00BXgHK7pVdAd43fnvd8vtGe3av7MruHdceFROlF1e8KEOGc1tWi1WhOUIzXJM7EK4AAABwVzLzcLd4hmEoxhGjaEe0omNv/DiiZY+1p9i24eQGLdi/QIYMWWRRrQK1VDhH4cRBKT4QxYcp+3VFO6JNqT81sntlV06fnM7Ak1QgCvAKUA7vHM52Z58b6/jafGWxWNL1+MPqDLs5m2axamjtoXfErJVEuAIAALijJLyKWuGgwlldjqTMCTHxYh2xzoBid9hTHWjWHV+n/+3/nzPI1C9cX2E5w1y3E7/dhNuJ30aC5dGx0boWdU1vz3jbtJBjyNC6E+ukE2lf1y+bn/yy+cnX5hv3O5vrb5flXjf7Odtv9Ltqv6qXVryUaJboxxY/ZmmYaV2qtWrmq6nZS2arXaN2HrOfpwbhCgAA4A4xd89cDV8zXA45NGneJA2tM1StS7VO1M9hOBTriFWskeDHcfO3w3AoxohJtp/DcCjGEZOqfutPrNf8ffOdIaZh0Ya6L9d9qQ4xUbFRSQaa+HVjjdgMj5shQ6uOrtKqo6sytqEkTiXKZskmL5uXvG3e8rZ6y9vmLS/rLfdtXrpmv6btZ7cnWr9FWAuVylXKNSQlE4Z8s/lmaEYoKZ46S5TfP79KeJVQfv/8WV1KmhCuAAAAkpHRGRnDMBQZG5nosK9rMdeSPAfGpf2W35eiLumfC/84t+2QQ0PXDNWHGz6UIcMlNGUVQ4aWHFqiJYeWZNpjxAeW5EKMt9Vb12Ou6+9zfydat2GRhioSWMRlnaTWv7XN6rBqze9r1OjhRvL38Xcu97J6yWa1paruk1dPKnxOuMvFHqwWq/pV7Zfls0R1CtbRkctHFJoj1COC1Z2McAUAQCbLzEOm0uterMkwDMUYMYkP+UrmMLDfjv2m2btnu8zIhOUMS/ZCAYkuGnCjLbNdi7mWpv42iy3uxxr322qxKps1m6wWa6JlNotNVqtV2Sw3lidot1lsumK/kmSIqV+ovgrlKJRiaPGx+cSFlITL44NLEm3ZrNlSNWuTXJAZWHNguvYtu92uf23/qkBAAXl5eaV5fUkKCQjR0NpDPXKWKCQgxCPquBsQrgAA6eaJb9A97XwUl8sc33gzldRhXJ5ek2EYiQ4viz90LMoepYuOizp+5bgsNovrIWiG6yFmyR2itvr4av3w7w/OINOkeBOVy1Mu0Tk2qTnv5naHoiU8tyQtzJiR8bZ63/b8l5QOCYuKidKwtcMSnR8zvvF45Q/If9vQFB+OrBarqYeUJRdi3qr9Vpa+JnhqkGGW6O5HuPJwnvjGhZpSxxNrkjzvjafkmWNFTSlLb2i43Zv0+DffLud6OJJ4Q57UG3ZHrFYfX61Zu2fJkKGJ8yaqRckWqpqvaqJ+ic4tSeqckiRqS+05KvHtkTGR2n9pv/O5O4y4w7jG/TVO2axZ819wjCNGR68cTVTTmM1jJCnZsUnN99Z8PP9jU2o0ZGjhgYVaeGChKdtLjs1iS/bQMnusXQcjDiZa56HQh1Q0sGjiIOTlJz9b0mEpvl9qDx+7LYsS/burHlI949tNJ08NMZLnBhlmie5uFsMw0vcRzl0sIiJCQUFBunTpkgIDA7Osjrl75mrYmmHOT/Hal26v/xT4T5bVI0nrTqxzvnExo6aY2Bht3rRZ91e7X9ls6XujYXZNZvDEmjy1rru9JjP28aRqalWylarmr5pkwHBHcIiMidSei3sS1ZnPL58kJb1+Gt6k485is9gkQ/KyebkeNma1ucyaOA89u+XQsmsx17T34t5E2/1Pgf+oQEAB1wCU1EUDbnMY2a1tCYPU7cJOcjMyi9osyvI3xkcvHfW4q6idvHrS40KMmex2uxYuXKimTZum+7BApI0njXlasgHhKgmeEK6SelEHgLvd7c7/SOlN+nX7de2P2J9om5XzVlYevzw3+yY8l+R255ok95gp9Eu4/YioCA36Y5DrYVyy6uMHP1Zu39zuHFqn85Hn9fLKlxMdWjb24bEK9g92PYzMGvd8b3s+jsWqmJiYDL0J8tQg44mHdEqe9abzXsGYu58njXlasgGHBXqowxGHkwxWJXOWVA7vHFlQkXQ5+nKSnyxmpCbDMHT+/Hnlzp07XceAZ0ZNGeWJNUmeWde9UFNG9/Hb1VQhbwXl8c2T4snoqT0x3SUk3OiXXHCIiIrQ4NWDE4WGzx/+XMH+wclvO4U36Rk5FyS5N+gfP/hxlr5Bj3JEJXqD3qhooyyrR0r60st1C9fNsno89dAyTz2sDIDnIlx5qCKBRWS1WBO9Sfiy4ZdZ9uKe3BuXjNTk/FSikbmfdt5t43S31nUv1JTRffx2NX364KdZuk/FGDGJ3gzXD62fZfXwBp2aMgPnxwBIC2tWF4Ckxb9JsFri/kSe8CaBmu7cmjy1Lmq6c2uS4t4ML2qzSBPCJ2hRm0UecbhU61KttaDFAvUM6KkFLRZ4RE1S3N+wRkiNLP+bJURNAGA+Zq48mCd+ikdNd25NUlxdNfPV9KgToT1xrKgp9TzxU/38/vlVwquE8vvnz+pSAAD3GMKVh/PENy7UlDqeWJPkmW88PXGsqAkAAKQVhwUCAAAAgAkIVwAAAABgAsIVAAAAAJiAcAUAAAAAJiBcAQAAAIAJCFcAAAAAYALCFQAAAACYgHAFAAAAACYgXAEAAACACQhXAAAAAGACwhUAAAAAmIBwBQAAAAAmIFwBAAAAgAkIVwAAAABgAsIVAAAAAJggW1YX4IkMw5AkRUREZHEldz+73a5r164pIiJCXl5eWV3OPYExdy/G2/0Yc/djzN2L8XY/xtz9PGnM4zNBfEa4HcJVEi5fvixJCg0NzeJKAAAAAHiCy5cvKygo6LZ9LEZqItg9xuFw6Pjx48qRI4csFktWl3NXi4iIUGhoqI4cOaLAwMCsLueewJi7F+Ptfoy5+zHm7sV4ux9j7n6eNOaGYejy5csqWLCgrNbbn1XFzFUSrFarChcunNVl3FMCAwOz/B/OvYYxdy/G2/0Yc/djzN2L8XY/xtz9PGXMU5qxiscFLQAAAADABIQrAAAAADAB4QpZysfHR0OHDpWPj09Wl3LPYMzdi/F2P8bc/Rhz92K83Y8xd787dcy5oAUAAAAAmICZKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAAAAAMAEhCtkmhEjRqhGjRrKkSOH8uXLp5YtW2r37t23XWfSpEmyWCwuP76+vm6q+M43bNiwRONXpkyZ264ze/ZslSlTRr6+vqpYsaIWLlzopmrvfMWKFUs03haLRX379k2yP/t32v32229q3ry5ChYsKIvFonnz5rksNwxDQ4YMUYECBeTn56eGDRtqz549KW537NixKlasmHx9fVWrVi1t2LAhk57Bned2Y2632zVw4EBVrFhRAQEBKliwoLp27arjx4/fdpvpeW26l6S0n3fv3j3R+D366KMpbpf9PGkpjXdSr+sWi0UfffRRsttkH09eat4PRkZGqm/fvsqTJ4+yZ8+uNm3a6NSpU7fdbnpf/zMb4QqZZtWqVerbt6/WrVunJUuWyG63q3Hjxrp69ept1wsMDNSJEyecP4cOHXJTxXeH8uXLu4zfH3/8kWzfNWvWqFOnTurVq5e2bNmili1bqmXLltqxY4cbK75zbdy40WWslyxZIklq165dsuuwf6fN1atXVblyZY0dOzbJ5SNHjtSYMWP01Vdfaf369QoICFB4eLgiIyOT3ebMmTM1YMAADR06VJs3b1blypUVHh6u06dPZ9bTuKPcbsyvXbumzZs366233tLmzZs1d+5c7d69W48//niK203La9O9JqX9XJIeffRRl/GbPn36bbfJfp68lMY74TifOHFCEyZMkMViUZs2bW67XfbxpKXm/WD//v31888/a/bs2Vq1apWOHz+u1q1b33a76Xn9dwsDcJPTp08bkoxVq1Yl22fixIlGUFCQ+4q6ywwdOtSoXLlyqvu3b9/eaNasmUtbrVq1jKefftrkyu4NL774ohEWFmY4HI4kl7N/Z4wk48cff3TedzgcRkhIiPHRRx852y5evGj4+PgY06dPT3Y7NWvWNPr27eu8HxsbaxQsWNAYMWJEptR9J7t1zJOyYcMGQ5Jx6NChZPuk9bXpXpbUmHfr1s1o0aJFmrbDfp46qdnHW7RoYTz88MO37cM+nnq3vh+8ePGi4eXlZcyePdvZZ9euXYYkY+3atUluI72v/+7AzBXc5tKlS5Kk3Llz37bflStXVLRoUYWGhqpFixb6+++/3VHeXWPPnj0qWLCgSpQooS5duujw4cPJ9l27dq0aNmzo0hYeHq61a9dmdpl3nejoaE2ZMkU9e/aUxWJJth/7t3kOHDigkydPuuzDQUFBqlWrVrL7cHR0tDZt2uSyjtVqVcOGDdnv0+nSpUuyWCzKmTPnbful5bUJia1cuVL58uVT6dKl9eyzz+rcuXPJ9mU/N8+pU6e0YMEC9erVK8W+7OOpc+v7wU2bNslut7vsr2XKlFGRIkWS3V/T8/rvLoQruIXD4dBLL72kBx54QBUqVEi2X+nSpTVhwgT99NNPmjJlihwOh+rUqaOjR4+6sdo7V61atTRp0iT9+uuv+vLLL3XgwAHVq1dPly9fTrL/yZMnlT9/fpe2/Pnz6+TJk+4o964yb948Xbx4Ud27d0+2D/u3ueL307Tsw2fPnlVsbCz7vUkiIyM1cOBAderUSYGBgcn2S+trE1w9+uij+u6777Rs2TJ9+OGHWrVqlZo0aaLY2Ngk+7Ofm2fy5MnKkSNHioeosY+nTlLvB0+ePClvb+9EH9Dcbn9Nz+u/u2TL0kfHPaNv377asWNHiscf165dW7Vr13ber1OnjsqWLauvv/5a77zzTmaXecdr0qSJ83alSpVUq1YtFS1aVLNmzUrVp25Iv/Hjx6tJkyYqWLBgsn3Yv3E3sdvtat++vQzD0Jdffnnbvrw2ZUzHjh2dtytWrKhKlSopLCxMK1eu1COPPJKFld39JkyYoC5duqR48SH28dRJ7fvBOxkzV8h0/fr10//+9z+tWLFChQsXTtO6Xl5eqlq1qvbu3ZtJ1d3dcubMqfvuuy/Z8QsJCUl0NZ5Tp04pJCTEHeXdNQ4dOqSlS5eqd+/eaVqP/Ttj4vfTtOzDefPmlc1mY7/PoPhgdejQIS1ZsuS2s1ZJSem1CbdXokQJ5c2bN9nxYz83x++//67du3en+bVdYh9PSnLvB0NCQhQdHa2LFy+69L/d/pqe1393IVwh0xiGoX79+unHH3/U8uXLVbx48TRvIzY2Vtu3b1eBAgUyocK735UrV7Rv375kx6927dpatmyZS9uSJUtcZleQsokTJypfvnxq1qxZmtZj/86Y4sWLKyQkxGUfjoiI0Pr165Pdh729vVWtWjWXdRwOh5YtW8Z+n0rxwWrPnj1aunSp8uTJk+ZtpPTahNs7evSozp07l+z4sZ+bY/z48apWrZoqV66c5nXZx29K6f1gtWrV5OXl5bK/7t69W4cPH052f03P67/bZOnlNHBXe/bZZ42goCBj5cqVxokTJ5w/165dc/Z58sknjddff915f/jw4caiRYuMffv2GZs2bTI6duxo+Pr6Gn///XdWPIU7zssvv2ysXLnSOHDggLF69WqjYcOGRt68eY3Tp08bhpF4vFevXm1ky5bN+Pjjj41du3YZQ4cONby8vIzt27dn1VO448TGxhpFihQxBg4cmGgZ+3fGXb582diyZYuxZcsWQ5IxatQoY8uWLc4r033wwQdGzpw5jZ9++sn466+/jBYtWhjFixc3rl+/7tzGww8/bHz++efO+zNmzDB8fHyMSZMmGTt37jSeeuopI2fOnMbJkyfd/vw80e3GPDo62nj88ceNwoULG1u3bnV5bY+KinJu49YxT+m16V53uzG/fPmy8corrxhr1641Dhw4YCxdutS4//77jVKlShmRkZHObbCfp15KryuGYRiXLl0y/P39jS+//DLJbbCPp15q3g8+88wzRpEiRYzly5cbf/75p1G7dm2jdu3aLtspXbq0MXfuXOf91Lz+ZwXCFTKNpCR/Jk6c6OzToEEDo1u3bs77L730klGkSBHD29vbyJ8/v9G0aVNj8+bN7i/+DtWhQwejQIEChre3t1GoUCGjQ4cOxt69e53Lbx1vwzCMWbNmGffdd5/h7e1tlC9f3liwYIGbq76zLVq0yJBk7N69O9Ey9u+MW7FiRZKvI/Hj6nA4jLfeesvInz+/4ePjYzzyyCOJ/hZFixY1hg4d6tL2+eefO/8WNWvWNNatW+emZ+T5bjfmBw4cSPa1fcWKFc5t3DrmKb023etuN+bXrl0zGjdubAQHBxteXl5G0aJFjT59+iQKSeznqZfS64phGMbXX39t+Pn5GRcvXkxyG+zjqZea94PXr183nnvuOSNXrlyGv7+/0apVK+PEiROJtpNwndS8/mcFi2EYRubMiQEAAADAvYNzrgAAAADABIQrAAAAADAB4QoAAAAATEC4AgAAAAATEK4AAAAAwASEKwAAAAAwAeEKAAAAAExAuAIAAAAAExCuAADwQMWKFdPo0aOzugwAQBoQrgAAHqN79+6yWCx65plnEi3r27evLBaLunfvnqk1TJo0SRaLRRaLRTabTbly5VKtWrX09ttv69KlS5nyeDlz5jR9uwAA9yNcAQA8SmhoqGbMmKHr16872yIjIzVt2jQVKVLELTUEBgbqxIkTOnr0qNasWaOnnnpK3333napUqaLjx4+7pQYAwJ2HcAUA8Cj333+/QkNDNXfuXGfb3LlzVaRIEVWtWtWl76+//qq6desqZ86cypMnjx577DHt27fPufy7775T9uzZtWfPHmfbc889pzJlyujatWvJ1mCxWBQSEqICBQqobNmy6tWrl9asWaMrV67otddec/ZzOBwaMWKEihcvLj8/P1WuXFk//PCDc/nKlStlsVi0YMECVapUSb6+vvrPf/6jHTt2OJf36NFDly5dcs6WDRs2zLn+tWvX1LNnT+XIkUNFihTRN998k/YBBQC4DeEKAOBxevbsqYkTJzrvT5gwQT169EjU7+rVqxowYID+/PNPLVu2TFarVa1atZLD4ZAkde3aVU2bNlWXLl0UExOjBQsW6Ntvv9XUqVPl7++fppry5cunLl26aP78+YqNjZUkjRgxQt99952++uor/f333+rfv7+eeOIJrVq1ymXdV199VZ988ok2btyo4OBgNW/eXHa7XXXq1NHo0aOdM2UnTpzQK6+84lzvk08+UfXq1bVlyxY999xzevbZZ7V79+401Q0AcJ9sWV0AAAC3euKJJzRo0CAdOnRIkrR69WrNmDFDK1eudOnXpk0bl/sTJkxQcHCwdu7cqQoVKkiSvv76a1WqVEkvvPCC5s6dq2HDhqlatWrpqqtMmTK6fPmyzp07p6CgIL3//vtaunSpateuLUkqUaKE/vjjD3399ddq0KCBc72hQ4eqUaNGkqTJkyercOHC+vHHH9W+fXsFBQU5Z8pu1bRpUz333HOSpIEDB+rTTz/VihUrVLp06XTVDwDIXIQrAIDHCQ4OVrNmzTRp0iQZhqFmzZopb968ifrt2bNHQ4YM0fr163X27FnnjNXhw4ed4SpXrlwaP368wsPDVadOHb3++uvprsswDElxhw3u3btX165dc4ameNHR0YkOX4wPX5KUO3dulS5dWrt27Urx8SpVquS8HR/ATp8+ne76AQCZi3AFAPBIPXv2VL9+/SRJY8eOTbJP8+bNVbRoUY0bN04FCxaUw+FQhQoVFB0d7dLvt99+k81m04kTJ3T16lXlyJEjXTXt2rVLgYGBypMnj/bv3y9JWrBggQoVKuTSz8fHJ13bv5WXl5fLfYvF4gyQAADPwzlXAACP9Oijjyo6Olp2u13h4eGJlp87d067d+/W4MGD9cgjj6hs2bK6cOFCon5r1qzRhx9+qJ9//lnZs2d3Bra0On36tKZNm6aWLVvKarWqXLly8vHx0eHDh1WyZEmXn9DQUJd1161b57x94cIF/fvvvypbtqwkydvb23kOFwDgzsbMFQDAI9lsNuehczabLdHyXLlyKU+ePPrmm29UoEABHT58ONEhf5cvX9aTTz6pF154QU2aNFHhwoVVo0YNNW/eXG3btk32sQ3D0MmTJ2UYhi5evKi1a9fq/fffV1BQkD744ANJUo4cOfTKK6+of//+cjgcqlu3ri5duqTVq1crMDBQ3bp1c27v7bffVp48eZQ/f369+eabyps3r1q2bCkp7suCr1y5omXLlqly5cry9/dP88U2AACegZkrAIDHCgwMVGBgYJLLrFarZsyYoU2bNqlChQrq37+/PvroI5c+L774ogICAvT+++9LkipWrKj3339fTz/9tI4dO5bs40ZERKhAgQIqVKiQateura+//lrdunXTli1bVKBAAWe/d955R2+99ZZGjBihsmXL6tFHH9WCBQtUvHhxl+198MEHevHFF1WtWjWdPHlSP//8s7y9vSVJderU0TPPPKMOHTooODhYI0eOTNdYAQCynsWIPzsXAACYauXKlXrooYd04cIF5cyZM6vLAQBkMmauAAAAAMAEhCsAAAAAMAGHBQIAAACACZi5AgAAAAATEK4AAAAAwASEKwAAAAAwwf/br2MBAAAAgEH+1pPYWRbJFQAAwECuAAAABnIFAAAwkCsAAICBXAEAAAwCYsjS0fUNu5gAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# set the number of trees ranging from 10 - 201, in an interval of 50 to make it more efficient.\n",
        "n_estimators_values = range(10, 201, 50)\n",
        "\n",
        "# Store accuracies for each algorithm\n",
        "\n",
        "macro_accuracies = []\n",
        "micro_accuracies = []\n",
        "overall_accuracies = []\n",
        "\n",
        "\n",
        "# Constants\n",
        "MAX_DEPTH = None\n",
        "MIN_SAMPLES_SPLIT = 2\n",
        "MIN_SAMPLES_LEAF = 1\n",
        "RANDOM_STATE = 42\n",
        "\n",
        "\n",
        "for n in n_estimators_values:\n",
        "    rf = RandomForestClassifier(n_estimators=n, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,\n",
        "                                min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE, n_jobs=-1)\n",
        "    rf.fit(X_train_pca, y_train)\n",
        "    y_pred = rf.predict(X_test_pca)\n",
        "\n",
        "\n",
        "    # ----* calcualte macro accuracy *---- #.\n",
        "    NUM_CLASSES = len(set(y_train))\n",
        "\n",
        "    conf_mat = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]\n",
        "    err_cnt = 0\n",
        "\n",
        "    # Use y_test for true labels, so the row represents the actual class.\n",
        "    for i, val in enumerate(y_test):\n",
        "    # Use y_pred for predicted labels, the column represent the predicted class.\n",
        "      conf_mat[val][y_pred[i]] += 1\n",
        "\n",
        "\n",
        "\n",
        "    c_mat = [[ 8,  5,  20],\n",
        "            [ 2, 10,  10],\n",
        "            [ 5,  5, 270]]\n",
        "\n",
        "\n",
        "    c_mat = np.array(conf_mat)\n",
        "\n",
        "    # get the statistics for each class.\n",
        "    tp = TP(c_mat)\n",
        "    fp = FP(c_mat)\n",
        "    tn = TN(c_mat)\n",
        "    fn = FN(c_mat)\n",
        "    acc = accuracy(tp, tn, fp, fn)\n",
        "    acc = [round(x, 3) for x in acc]\n",
        "\n",
        "    macro_acc = macro_ACCURACY(acc)\n",
        "    macro_accuracies.append(macro_acc)\n",
        "\n",
        "\n",
        "    # ----* calcualte micro accuracy *---- #.\n",
        "    micro_c_mat = micro_conf_mat(tp, tn, fp, fn)\n",
        "\n",
        "    for i in range(2):\n",
        "      for j in range(2):\n",
        "        micro_c_mat[i][j] = round(micro_c_mat[i][j], 3)\n",
        "\n",
        "    micro_acc = micro_ACCURACY(micro_c_mat)\n",
        "    micro_accuracies.append(micro_acc)\n",
        "\n",
        "\n",
        "     # ----* calcualte overall accuracy *---- #.\n",
        "    total_cnt = 0\n",
        "    total_correct = 0\n",
        "    for i in range(NUM_CLASSES):\n",
        "      for j in range(NUM_CLASSES):\n",
        "        total_cnt += c_mat[i][j]\n",
        "        if i==j:\n",
        "          total_correct += c_mat[i][j]\n",
        "\n",
        "    overall_acc = total_correct/total_cnt\n",
        "    overall_accuracies.append(overall_acc)\n",
        "\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.plot(n_estimators_values, overall_accuracies, marker='o', label='Overall Accuracy')\n",
        "plt.plot(n_estimators_values, micro_accuracies, marker='x', label='Micro Accuracy')\n",
        "plt.plot(n_estimators_values, macro_accuracies, marker='.', label='Macro Accuracy')\n",
        "plt.xlabel('Number of Trees (N_ESTIMATORS)')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('Random Forest Accuracy with varying N_ESTIMATORS')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "WYl5ddc5BoiV",
        "outputId": "77f7fe2f-1e38-44b8-b340-ba941781a854"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA18AAAIjCAYAAAD80aFnAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACiA0lEQVR4nOzdeVwU9f8H8NfuAruAgMopiqJomreh8NW8E1GURMn71jRLOqRM8UItxSxNLdMyUFNR88gyzUQ88/55Zt5XmIpnioLAsju/P3Anll1gF2EG9PV8PPYB+5nPfOYz792Fee985jMKQRAEEBERERERUbFSyt0BIiIiIiKiFwGTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iKnaDBg2Cr6+v3N0gKnZLliyBQqHA1atXLa77f//3f8XfMYm0bt0arVu3lrsbREQlFpMvoueI4WDO8LCxsUHFihUxaNAgXL9+Xe7ulRi545TzMXbsWLm7Z9b06dOxYcMGq9c7c+YMFAoFNBoNHjx4UOT9ooJ98803WLJkidzdeO75+vpCoVDg3XffNVm2c+dOKBQKrF271qo28/o7oVAoMGLECKO6GzduRKtWreDh4QEHBwdUq1YNPXr0wJYtWwBkJ6b5tWd4TJ48Wdyfzp07m+3Pm2++aba/48ePF+vcvXvXbJ0ePXpAoVBgzJgxYtnVq1ct6lvuLxaSkpIwYsQI+Pr6Qq1Ww8PDA2FhYdi7d6/Jdg2vgeGhUqng4eGBN954A2fOnDHb14JiSlQa2cjdASIqelOnTkXVqlWRnp6OAwcOYMmSJfjjjz9w6tQpaDQaubtXYhjilFPdunVl6k3+pk+fjjfeeANhYWFWrbd8+XJ4eXnh33//xdq1a/M8aKOi0b9/f/Tq1QtqtVos++abb+Dm5oZBgwbJ1zGJbN26Ve4uYNGiRYiKioK3t3eRtBcUFIQBAwaYlL/00kvi71988QVGjx6NVq1aISoqCg4ODrh48SK2bduGVatWoUOHDhg/frzR5+/w4cOYN28exo0bh5dfflksr1+/fr790Wg0WLduHb755hvY2dkZLVu5ciU0Gg3S09PNrpuSkoKNGzfC19cXK1euxIwZM6BQKODu7o5ly5YZ1Z01axb++ecffPnll0bl7u7uAIC9e/ciJCQEAPDmm2+idu3aSE5OxpIlS9CiRQvMnTvXbCL83nvvoUmTJtBqtTh58iQWLlyInTt34tSpU/Dy8rIqpkSlkkBEz43FixcLAITDhw8blY8ZM0YAIKxevVqWfg0cOFCoUqWKLNs2J684FZXHjx8XeZuOjo7CwIEDrVpHr9cLvr6+QmRkpNC1a1ehdevWRd6volIcMSsp6tSpI7Rq1cqkvLjfh0WhNL0uVapUEerUqSPY2NgI7777rtGyHTt2CACENWvWWNUmAGHkyJH51tFqtYKzs7MQFBRkdvmtW7fMlq9Zs0YAIOzYscPs8ipVqgidOnUy6U9YWJigVCqFDRs2GC3bu3evAEAIDw8XAAh37twxaTMuLk6wtbUVtm/fLgAQdu7cmed+derUKc+/2/fv3xe8vLwET09P4eLFi0bL0tLShBYtWghKpVLYu3evWJ7Xa7BgwQIBgPDZZ5+JZYWNKVFpwGGHRC+AFi1aAAAuXboklmVmZmLSpEnw9/eHi4sLHB0d0aJFC+zYscNoXcNwlC+++ALfffcd/Pz8oFar0aRJExw+fNhkWxs2bEDdunWh0WhQt25d/PTTT2b7lJqaig8//BA+Pj5Qq9WoWbMmvvjiCwiCYFRPoVAgIiICa9asQe3atWFvb4+mTZvizz//BAB8++23qF69OjQaDVq3bm3RtTaW2r59O1q0aAFHR0eULVsWXbp0MRkeM3nyZCgUCpw+fRp9+vRBuXLl0Lx5c3H58uXL4e/vD3t7e5QvXx69evXCtWvXjNq4cOECwsPD4eXlBY1Gg0qVKqFXr154+PChGIPU1FQsXbpUHLJjyVmUvXv34urVq+jVqxd69eqF3bt3459//jGpp9frMXfuXNSrVw8ajQbu7u7o0KGDybVIy5cvR0BAABwcHFCuXDm0bNnS6ExHziFTOfn6+hr11zDsc9euXXjnnXfg4eGBSpUqAQD+/vtvvPPOO6hZsybs7e3h6uqK7t27m31dHzx4gFGjRolDnipVqoQBAwbg7t27ePz4MRwdHfH++++brPfPP/9ApVIhJiYmz9i98sor6Natm1FZvXr1oFAocPLkSbFs9erVUCgU4vsi9zVfvr6++Ouvv7Br1y7xtct9TVRGRgYiIyPh7u4OR0dHdO3aFXfu3Mmzb0D2WQGFQoG///7bZFlUVBTs7Ozw77//AgD27NmD7t27o3LlylCr1fDx8cGoUaPw5MkTo/UGDRqEMmXK4NKlSwgJCYGTkxP69u2L6Oho2Nramu3T8OHDUbZsWfFMS+5rvgxDzX788UdMmzYNlSpVgkajwWuvvYaLFy+atDd//nxUq1YN9vb2CAgIwJ49e6y6jszX1xcDBgzAokWLcOPGDYvWeVZ3795FSkoKXn31VbPLPTw8inR7FStWRMuWLREfH29UvmLFCtSrVy/fs/crVqxAUFAQ2rRpg5dffhkrVqwoVB++/fZbJCcn4/PPP4efn5/RMnt7e/Fv1dSpUwtsy9z/J6ljSiQlJl9ELwDDgWC5cuXEspSUFHz//fdo3bo1PvvsM0yePBl37txBcHAwjh8/btJGfHw8Pv/8c7z11lv49NNPcfXqVXTr1g1arVass3XrVoSHh0OhUCAmJgZhYWEYPHiwyUG8IAh4/fXX8eWXX6JDhw6YPXs2atasidGjRyMyMtJk23v27MGHH36IgQMHYvLkyThz5gw6d+6M+fPnY968eXjnnXcwevRo7N+/H0OGDLE4Lg8fPsTdu3eNHgbbtm1DcHAwbt++jcmTJyMyMhL79u3Dq6++ajYR6N69O9LS0jB9+nQMGzYMADBt2jQMGDAANWrUwOzZs/HBBx8gMTERLVu2FK+/yszMRHBwMA4cOIB3330X8+fPx/Dhw3H58mWxzrJly6BWq9GiRQssW7YMy5Ytw1tvvVXg/q1YsQJ+fn5o0qQJQkND4eDggJUrV5rUGzp0KD744AP4+Pjgs88+w9ixY6HRaHDgwAGxzpQpU9C/f3/Y2tpi6tSpmDJlCnx8fLB9+3aL453bO++8g9OnT2PSpEnitXaHDx/Gvn370KtXL8ybNw8jRoxAYmIiWrdujbS0NHHdx48fo0WLFvjqq6/Qvn17zJ07FyNGjMDZs2fxzz//oEyZMujatStWr14NnU5ntN2VK1dCEAT07ds3z761aNECf/zxh/j8/v37+Ouvv6BUKrFnzx6xfM+ePXB3dzcaNpbTnDlzUKlSJdSqVUt87caPH29U591338WJEycQHR2Nt99+Gxs3bkRERES+sTNct/Pjjz+aLPvxxx/Rvn178fO+Zs0apKWl4e2338ZXX32F4OBgfPXVV2aH0mVlZSE4OBgeHh744osvEB4ejv79+yMrKwurV682qpuZmYm1a9ciPDy8wOHMM2bMwE8//YSPPvoIUVFROHDggEn8FyxYgIiICFSqVAkzZ85EixYtEBYWZvYLg/yMHz8eWVlZmDFjhlXr5SU9Pd3k78Tdu3eRmZkJIDsRsLe3x8aNG3H//v0i2WZB+vTpg40bN+Lx48cAsl+3NWvWoE+fPnmuc+PGDezYsQO9e/cGAPTu3Rtr164V98MaGzduhEajQY8ePcwur1q1Kpo3b47t27ebJPm5mfv/JEdMiSQj85k3IipChmFM27ZtE+7cuSNcu3ZNWLt2reDu7i6o1Wrh2rVrYt2srCwhIyPDaP1///1X8PT0FIYMGSKWXblyRQAguLq6Cvfv3xfLf/75ZwGAsHHjRrGsYcOGQoUKFYQHDx6IZVu3bhUAGA1f2bBhgwBA+PTTT422/8YbbwgKhcJoGAsAQa1WC1euXBHLvv32WwGA4OXlJaSkpIjlUVFRAgCjuvnFydwj5754eHgI9+7dE8tOnDghKJVKYcCAAWJZdHS0AEDo3bu30TauXr0qqFQqYdq0aUblf/75p2BjYyOWHzt2zKLhUNYOO8zMzBRcXV2F8ePHi2V9+vQRGjRoYFTPMPzovffeM2lDr9cLgiAIFy5cEJRKpdC1a1dBp9OZrSMI2a9VdHS0STtVqlQx6rsh/s2bNxeysrKM6qalpZmsv3//fgGA8MMPP4hlkyZNEgAI69evz7Pfv//+uwBA+O2334yW169f3+wwwJwMQ8JOnz4tCIIg/PLLL4JarRZef/11oWfPnkZtde3a1WTfcr4HCxp22K5dO6M4jho1SlCpVEafI3OaNm0q+Pv7G5UdOnTIJFbmYhoTEyMoFArh77//FssGDhwoABDGjh1rdluBgYFGZevXrzcZNteqVSujfTUMNXv55ZeN/t7MnTtXACD8+eefgiAIQkZGhuDq6io0adJE0Gq1Yr0lS5YIAAp8vQTBeJje4MGDBY1GI9y4ccOoH4UZdpjXY+XKlWI9w/vR0dFR6NixozBt2jThyJEj+bZd2GGHI0eOFO7fvy/Y2dkJy5YtEwRBEDZt2iQoFArh6tWr4t+k3MMOv/jiC8He3l78m3n+/HkBgPDTTz+Z3X5+ww7Lli1r8rckt/fee08AIJw8eVIQhP9eg7i4OOHOnTvCjRs3hC1btgjVq1cXFAqFcOjQIaP1CxNTotKAZ76InkPt2rWDu7s7fHx88MYbb8DR0RG//PKLOLQLAFQqlXixtl6vx/3795GVlYXGjRvj6NGjJm327NnT6JtJw1CRy5cvAwBu3ryJ48ePY+DAgXBxcRHrBQUFoXbt2kZtbd68GSqVCu+9955R+YcffghBEPDbb78Zlb/22mtGU9UHBgYCAMLDw+Hk5GRSbuhTQebPn4+EhASjR859GTRoEMqXLy/Wr1+/PoKCgrB582aTtnLPfLZ+/Xro9Xr06NHD6NtyLy8v1KhRQxzeaYjV77//bnRm51n99ttvuHfvnvgtN5D9TfeJEyfw119/iWXr1q2DQqFAdHS0SRsKhQJA9lBSvV6PSZMmQalUmq1TGMOGDYNKpTIqs7e3F3/XarW4d+8eqlevjrJlyxq9L9etW4cGDRqga9euefa7Xbt28Pb2NhpaderUKZw8eRL9+vXLt2+G9/fu3bsBZJ/hatKkCYKCgsQzXw8ePMCpU6fEuoU1fPhwozi2aNECOp3O7JDCnHr27IkjR44YDddavXo11Go1unTpIpbljGlqairu3r2LZs2aQRAEHDt2zKTdt99+26RswIABOHjwoNG2VqxYAR8fH7Rq1arAfRw8eLDR5BC5/3783//9H+7du4dhw4bBxua/ucD69u1r9HfHUhMmTCiys19dunQx+TuRkJCANm3aiHWmTJmC+Ph4NGrUCL///jvGjx8Pf39/vPLKK3nO5PcsypUrhw4dOohnsuPj49GsWTNUqVIlz3VWrFiBTp06iX8za9SoAX9//0INPXz06JHR315zDMtTUlKMyocMGQJ3d3d4e3ujQ4cOePjwIZYtW4YmTZoY1ZM6pkRSYfJF9BwyJBVr165FSEgI7t69azT7msHSpUtRv359aDQauLq6wt3dHZs2bRKvNcqpcuXKRs8NB0SG60oMB4o1atQwWbdmzZpGz//++294e3ub/PM2DN3KfdCZe9uGhMXHx8dsuaFPBQkICEC7du2MHjm3n7vfhj7evXsXqampRuW5Z028cOECBEFAjRo14O7ubvQ4c+YMbt++La4XGRmJ77//Hm5ubggODsb8+fPNvgbWWL58OapWrQq1Wo2LFy/i4sWL8PPzg4ODg9HB1qVLl+Dt7W2UZOZ26dIlKJVKkyT6WeWOGQA8efIEkyZNEq8FdHNzg7u7Ox48eGAUk0uXLhU4M6VSqUTfvn2xYcMGMbFdsWIFNBoNunfvnu+6np6eqFGjhpho7dmzBy1atEDLli1x48YNXL58GXv37oVer3/m5Kugz1ZeunfvDqVSKQ4HFAQBa9asQceOHeHs7CzWS0pKEr9IKFOmDNzd3cWEKff7zMbGxuhLGoOePXtCrVaL752HDx/i119/Rd++fS1KwC39+1G9enWT/hTmHoHVqlVD//798d133+HmzZtWr59TpUqVTP5OtGvXDp6enkb1evfujT179uDff//F1q1b0adPHxw7dgyhoaF5zj74LPr06YOEhAQkJSVhw4YN+Q45PHPmDI4dO4ZXX31V/Htw8eJFtG7dGr/++qtJglQQJycnPHr0KN86huW5/85PmjQJCQkJ+OmnnzBgwAA8fPjQ5EsdA6ljSiQFJl9EzyFDUhEeHo5ffvkFdevWRZ8+fcTrA4Dsg/NBgwbBz88PsbGx2LJlCxISEtC2bVvo9XqTNnOfoTAQck2QURzy2racfcot59kFIPtsokKhEOOa+/Htt9+KdWfNmoWTJ09i3LhxePLkCd577z3UqVPH6mtdDAzTSV+5cgU1atQQH7Vr10ZaWhri4+MljVHua64McscMyL7+adq0aejRowd+/PFHbN26FQkJCXB1dTX7vizIgAED8PjxY2zYsAGCICA+Ph6dO3c2Ojubl+bNm2PPnj148uQJjhw5ghYtWqBu3booW7Ys9uzZgz179qBMmTJo1KiR1f3KqbDvY29vb7Ro0UK87uvAgQNISkpCz549xTo6nQ5BQUHYtGkTxowZgw0bNiAhIUG871jumKrVarMHwuXKlUPnzp3F5Gvt2rXIyMgo8Azis+7jszBc+/XZZ58V2zbMcXZ2RlBQEFasWIGBAwfi0qVLOHjwYJFv5/XXX4darcbAgQORkZGR5/VXQPbfewAYNWqU0d+EWbNmIT09HevWrbNq2y+//DLOnTuHjIyMPOucPHkStra2Jl/I1atXD+3atUNYWBiWLl2K119/HcOGDTOZiCgnqWJKJAUmX0TPOcOsbjdu3MDXX38tlq9duxbVqlXD+vXr0b9/fwQHB6Ndu3aF/jbRMNzlwoULJsvOnTtnUvfGjRsm35yePXvWqC25GLafu99Adh/d3Nzg6OiYbxt+fn4QBAFVq1Y1+635//73P6P69erVw4QJE7B7927s2bMH169fx8KFC8Xl1gzvW79+PdLT07FgwQKsWbPG6PHpp5/i77//Fm+C6ufnhxs3buR7Ubufnx/0ej1Onz6d73bLlStnciPnzMxMq848rF27FgMHDsSsWbPwxhtvICgoCM2bNzdp18/PD6dOnSqwvbp166JRo0ZYsWIF9uzZg6SkJPTv39+ivrRo0QJJSUlYtWoVdDodmjVrBqVSKSZle/bsQbNmzfJMLAyeZWhmQXr27IkTJ07g3LlzWL16NRwcHBAaGiou//PPP3H+/HnMmjULY8aMQZcuXcThmNYaMGAAzp8/j8OHD2PFihVo1KgR6tSpUyT7YfjM5Z4BMSsrq9AzmPr5+aFfv3749ttvn/nsV2E1btwYAIpl+/b29ggLC8POnTsRFBQENzc3s/UMXzq0adPG5O/BmjVrUL9+fauHHnbu3Bnp6elYs2aN2eVXr17Fnj170LZtW7NfsuQ0Y8YMpKenY9q0aRZtuzhjSiQFJl9EL4DWrVsjICAAc+bMEZMrwwFjzm+eDx48iP379xdqGxUqVEDDhg2xdOlSo6FMCQkJJgftISEh0Ol0RskgAHz55ZdQKBTo2LFjofpQVHLuS86D/lOnTmHr1q3ijUXz061bN6hUKkyZMsXk231BEHDv3j0A2WepsrKyjJbXq1cPSqXS6FtlR0dHkwQkL8uXL0e1atUwYsQIvPHGG0aPjz76CGXKlBEPtsLDwyEIAqZMmWLSjqHfYWFhUCqVmDp1qsmZkpz75ufnJ14jZfDdd9/leebLHJVKZRKvr776yqSN8PBwnDhxwuytDHKv379/f2zduhVz5syBq6urxe8vw3DCzz77DPXr1xfPlrVo0QKJiYn4v//7P4uGHFrz2lkrPDwcKpUKK1euxJo1a9C5c2ejLwbMfc4FQcDcuXOt3lbHjh3h5uaGzz77DLt27bL4rJclGjduDFdXVyxatMjo87BixQqLhxGbM2HCBGi1WsycObMoumlWWlpann83DdevmhvCXBQ++ugjREdHY+LEiXnWMdxyYvDgwSZ/D9544w307NkTO3bssGpq/rfeegseHh4YPXq0yTW26enpGDx4MARBwKRJkwpsy8/PD+Hh4ViyZAmSk5MByBtTouJmU3AVInoejB49Gt27d8eSJUswYsQIdO7cGevXr0fXrl3RqVMnXLlyBQsXLkTt2rWNhidaIyYmBp06dULz5s0xZMgQ3L9/H1999RXq1Klj1GZoaCjatGmD8ePH4+rVq2jQoAG2bt2Kn3/+GR988IHJfWPk8Pnnn6Njx45o2rQphg4diidPnuCrr76Ci4uL2XtZ5ebn54dPP/0UUVFRuHr1KsLCwuDk5IQrV67gp59+wvDhw/HRRx9h+/btiIiIQPfu3fHSSy8hKysLy5Ytg0qlQnh4uNiev78/tm3bhtmzZ8Pb2xtVq1YVJxjJyTCddO7JTAzUajWCg4OxZs0azJs3D23atEH//v0xb948XLhwAR06dIBer8eePXvQpk0bREREoHr16hg/fjw++eQTtGjRAt26dYNarcbhw4fh7e0t3i/rzTffxIgRIxAeHo6goCCcOHECv//+e57fyJvTuXNnLFu2DC4uLqhduzb279+Pbdu2wdXV1aje6NGjsXbtWnTv3h1DhgyBv78/7t+/j19++QULFy5EgwYNxLp9+vTBxx9/jJ9++glvv/02bG1tLepL9erV4eXlhXPnzuHdd98Vy1u2bIkxY8YAgEXJl7+/PxYsWIBPP/0U1atXh4eHB9q2bWtRHwri4eGBNm3aYPbs2Xj06JHRkEMAqFWrFvz8/PDRRx/h+vXrcHZ2xrp16wqV0Nja2qJXr174+uuvoVKpjCZzeVZ2dnaYPHky3n33XbRt2xY9evTA1atXsWTJEvj5+RX67KHh7NfSpUsL3bfz58+Lw/Zy8vT0RFBQENLS0tCsWTP873//Q4cOHeDj44MHDx5gw4YN2LNnD8LCwp55aGpeGjRoYPReN2fFihVQqVTo1KmT2eWvv/46xo8fj1WrVpm91Yc5rq6uWLt2LTp16oRXXnkFb775JmrXro3k5GQsWbIEFy9exNy5c9GsWTOL2hs9ejR+/PFHzJkzBzNmzJA1pkTFTsKZFYmomBmmrj58+LDJMp1OJ/j5+Ql+fn5CVlaWoNfrhenTpwtVqlQR1Gq10KhRI+HXX38VBg4caDS9sGGq+c8//9ykTZiZWnzdunXCyy+/LKjVaqF27drC+vXrTdoUBEF49OiRMGrUKMHb21uwtbUVatSoIXz++edGU24btjFy5Eijsrz6ZOl00vnFKadt27YJr776qmBvby84OzsLoaGh4tTjBnlN62ywbt06oXnz5oKjo6Pg6Ogo1KpVSxg5cqRw7tw5QRAE4fLly8KQIUMEPz8/QaPRCOXLlxfatGkjbNu2zaids2fPCi1bthTs7e0FAHlOOz9r1iwBgJCYmJjnfhmm7/75558FQci+7cDnn38u1KpVS7CzsxPc3d2Fjh07mkzrHBcXJzRq1EhQq9VCuXLlhFatWgkJCQnicp1OJ4wZM0Zwc3MTHBwchODgYOHixYt5TjVvLv7//vuvMHjwYMHNzU0oU6aMEBwcLJw9e9akDUEQhHv37gkRERFCxYoVBTs7O6FSpUrCwIEDhbt375q0GxISIgAQ9u3bl2dczOnevbsAQFi9erVYlpmZKTg4OAh2dnbCkydPjOqbm2o+OTlZ6NSpk+Dk5GQ0bXpecTC8j/Oagjy3RYsWCQAEJycnk/4IgiCcPn1aaNeunVCmTBnBzc1NGDZsmHDixAkBgLB48WKx3sCBAwVHR8d8t2WYyr59+/Zml+c11Xzuz6ThM5xz+4IgCPPmzRP/JgUEBAh79+4V/P39hQ4dOuQfBMH81OyCkH2rBJVKVeRTzRv2U6vVCosWLRLCwsLEvjs4OAiNGjUSPv/8c5Nbehg8y1Tz+cn5N8lwy4kWLVrku07VqlWFRo0aGZXlN9W8wZUrV4Rhw4YJlStXFmxtbQU3Nzfh9ddfF/bs2WNSt6C/z61btxacnZ2FBw8eFDqmRKWBQhBkuDKdiIhIQl27dsWff/5pck0RWefEiRNo2LAhfvjhB4uvnXsWer0e7u7u6NatGxYtWlTs2yMiKm685ouIiJ5rN2/exKZNmyRJFp53ixYtQpkyZdCtW7cibzs9Pd3ker0ffvgB9+/fR+vWrYt8e0REcuA1X0RE9Fy6cuUK9u7di++//x62trZ466235O5SqbVx40acPn0a3333HSIiIgqc7bMwDhw4gFGjRqF79+5wdXXF0aNHERsbi7p16xZ4XzZr6HQ63LlzJ986ZcqUQZkyZYpsm0REBky+iIjoubRr1y4MHjwYlStXxtKlS+Hl5SV3l0qtd999F7du3UJISIjZmTGLgq+vL3x8fDBv3jzcv38f5cuXx4ABAzBjxgzY2dkV2XauXbtm9gbfOUVHR1s0sQ4RkbV4zRcRERG9MNLT0/HHH3/kW6datWqoVq2aRD0iohcJky8iIiIiIiIJcMINIiIiIiIiCfCar0LS6/W4ceMGnJycCn3zRyIiIiIiKv0EQcCjR4/g7e0NpTLv81tMvgrpxo0b8PHxkbsbRERERERUQly7dg2VKlXKczmTr0JycnICkB1gZ2dnmXtTemm1WmzduhXt27eHra2t3N15YTDu8mDc5cG4y4NxlwfjLg/GXR4lKe4pKSnw8fERc4S8MPkqJMNQQ2dnZyZfz0Cr1cLBwQHOzs6yf2heJIy7PBh3eTDu8mDc5cG4y4Nxl0dJjHtBlyNxwg0iIiIiIiIJyJ58zZ8/H76+vtBoNAgMDMShQ4fyrKvVajF16lT4+flBo9GgQYMG2LJli1GdmJgYNGnSBE5OTvDw8EBYWBjOnTtnVKd169ZQKBRGjxEjRhTL/hEREREREQEyJ1+rV69GZGQkoqOjcfToUTRo0ADBwcG4ffu22foTJkzAt99+i6+++gqnT5/GiBEj0LVrVxw7dkyss2vXLowcORIHDhxAQkICtFot2rdvj9TUVKO2hg0bhps3b4qPmTNnFuu+EhERERHRi03Wa75mz56NYcOGYfDgwQCAhQsXYtOmTYiLi8PYsWNN6i9btgzjx49HSEgIAODtt9/Gtm3bMGvWLCxfvhwATM6ELVmyBB4eHjhy5Ahatmwpljs4OMDLy8vivmZkZCAjI0N8npKSAiD7bJxWq7W4HTJmiB1jKC3GXR6MuzwYd3kw7vJg3OXBuMujJMXd0j7IlnxlZmbiyJEjiIqKEsuUSiXatWuH/fv3m10nIyMDGo3GqMze3h5//PFHntt5+PAhAKB8+fJG5StWrMDy5cvh5eWF0NBQTJw4EQ4ODnm2ExMTgylTppiUb926Nd/1yDIJCQlyd+GFxLjLg3GXB+MuD8ZdHoy7PBh3eZSEuKelpVlUT7bk6+7du9DpdPD09DQq9/T0xNmzZ82uExwcjNmzZ6Nly5bw8/NDYmIi1q9fD51OZ7a+Xq/HBx98gFdffRV169YVy/v06YMqVarA29sbJ0+exJgxY3Du3DmsX78+z/5GRUUhMjJSfG6YTrJ9+/ac7fAZaLVaJCQkICgoqMTMUvMiYNzlwbjLg3GXB+MuD8ZdHoy7PEpS3A2j4gpSqqaanzt3LoYNG4ZatWpBoVDAz88PgwcPRlxcnNn6I0eOxKlTp0zOjA0fPlz8vV69eqhQoQJee+01XLp0CX5+fmbbUqvVUKvVJuW2trayv9jPA8ZRHoy7PBh3eTDu8mDc5cG4y4Nxl0dJiLul25dtwg03NzeoVCrcunXLqPzWrVt5Xovl7u6ODRs2IDU1FX///TfOnj2LMmXKoFq1aiZ1IyIi8Ouvv2LHjh353mUaAAIDAwEAFy9eLOTeEBERERER5U+25MvOzg7+/v5ITEwUy/R6PRITE9G0adN819VoNKhYsSKysrKwbt06dOnSRVwmCAIiIiLw008/Yfv27ahatWqBfTl+/DgAoEKFCoXbGSIiIiIiogLIOuwwMjISAwcOROPGjREQEIA5c+YgNTVVnP1wwIABqFixImJiYgAABw8exPXr19GwYUNcv34dkydPhl6vx8cffyy2OXLkSMTHx+Pnn3+Gk5MTkpOTAQAuLi6wt7fHpUuXEB8fj5CQELi6uuLkyZMYNWoUWrZsifr160sfBCIiIiIieiHImnz17NkTd+7cwaRJk5CcnIyGDRtiy5Yt4iQcSUlJUCr/OzmXnp6OCRMm4PLlyyhTpgxCQkKwbNkylC1bVqyzYMECANk3Us5p8eLFGDRoEOzs7LBt2zYx0fPx8UF4eDgmTJhQ7PtLREREREQvLtkn3IiIiEBERITZZTt37jR63qpVK5w+fTrf9gRByHe5j48Pdu3aZVUfiYiIiIiInpVs13wRERERERG9SJh8ERERERERSYDJFxEREdGOGGDXTPPLds3MXk5E8ivln1UmX0RERERKFbBjmulB3a6Z2eVKlTz9IiJjpfyzKvuEG0REz7UdMdn/CFp9bLps10xArwPaREnfLyIyZviM7pj233PDwVyb8eY/w1Q0BCH7kf3k6e85fhrqmCwrqD6KoA1z9XMvA5ClhdOT68Cds4BKZeU2kXf9Z+o3rKxfmH23to95lVnRho0a8GsL7JgG1dW9cFc2hnLPaWD3jFLxWWXyRURUnAzf0AFAs1H/lec8qCPpCAIg6HP8I9cb/1MX9Hn8nvugIK82crdnQdv5rlfU/ctnvXz7ZP1+KXVZqJF8Fsq95wClIp/9kqd/xu3hvzZcq2d/NndMz15evhqQdABY1vW/dg3vpcIePFp0wIqC65spsxEEvJaaCpsrk57WydGWVX3Mr9/W1s+n7DlhC6AtAJyVuSMvGOWVnWiKXVBcEkpF4gUw+SKiF5VeD+izAL02+6cu6+lzQ5kO0GmNn+uzcpRl5Xquy9GW9r/nNmqgWuvsb+gu7cDLGa5Q/bAAuLYfqPw/IOMRsHWC5QeMkh6oG36HFQexhvVQQNv59Mnig2xY1CcbCOgi6IFjxfuWImMqALUB4KbMHSm0p+/9+5ezH6WEAkAZAMiUuSMlngJQKLJ/Av/9bvQTZspy18+uIwDQarWwtVNDYXUbOdsyV9+aNnLWf5b9zKvsWWKVuyz3Muv3Uzi5GgpBD0FlB0UpSLwAJl9ElJuYlJhLQswlHbnLzCUhxZXkWNNWrn7J8I2rMmkfXspZkHQg+0HFSlFwlUI0qoTxgYAyj99zHSzkXC/fNixcT/wdBfQjd9tPL/kusB+F3xe9IODaP//Ax6cylCpVPv0w1was3JenMbB4X5Q5tpNr2+e3Auc2AUqb7L8XtToDtTrlajf3gSLMlFl28Jh3fXPLCi7L0umwb/8BNGvWFDY2trnqo4BtFqbfsLK+Jfturq2iiG3O16loZWm1+G3zZoSEhMDW1rZYtkG57JoJhaCHTmEDlS4ze0RJKUjAmHyVVryORHqCUPgkxChRsCRxeNYkJ/+2bHRaBD9Jhc25Uab9EvRyR1o+CmX2AZfSBlDaZn/GVLY5yp4+VE+XKXMsUxW8nnB0afY3dAolFIFv533wV+ABOnL8bsGBbM4DbYsPZHOtV+zJRs79siYeebX3X11tVhYSt+/Aa+3awdbW7tniUUwHbs8jnVaL45s3wzskBMrScjC6a2Z24mUYvmQYHlyhQak4qAMAQavFv3/eg1ApACgtcSey1tPPpq7lWPz6qDY6O52GKuf1miUYk6/SKud1JDnfZHJcRyIIBSQJeZ2t0EKRmQH3lJNQXFBlH3hZlITkMbzLoiTnGdp6jpISBQANAGRZsUa+SYjNf4mIURJiYz4RUeVYJq5XUEKTu62CkiNrEyYbQFmME8Dm/IZOyALsy5b4fxDPDa0WGbYugKM7D0Ypb+Ym1zA3CQcRySvHZ1XfbBSweTP0LT6CSpXHsXEJw+SrtMr5DyHpAODbHLi8E7iyC6jSDMh8DGwZZ5yEWDW8K6+hYmbaEnSF3g0bAM0A4FIRxEQueSUPqpwH/7kThRxJi9nEJK+EpjBtmSY5Wr2AP/YeQPNWbWBrp7EgyeFdKZ5JKf6GjuiFodfB7AX7huf6wv+vI6IilPOzqtX+V15KPqtMvkqzVh8Dukxg9+fApcT/yv/el/2QW35nGJ4+FxQqPHycBuey5aG0sSuy4V2FPvNhdZJTsu8lkSetFikONwD3mjwTUNxK+Td0RC+M/Ibq8zNKVHKU8s8qk6/SrtUYYM+s7CFxCiXQeIgVw7ssHSpWiCTHwmsjsrRa7Hp6gWqpuSaAyBql/Bs6IiIiKjpMvkq7P77MTrxUdtlnwcp4loqsn+iFUcq/oSMiIqKiwws5SrOcFwdPvJP9c8e07HIiIiIiIipReOartOKsTEREREREpQqTr9KKszIREREREZUqTL5KK15HQkRERERUqvCaLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJyJ58zZ8/H76+vtBoNAgMDMShQ4fyrKvVajF16lT4+flBo9GgQYMG2LJli9VtpqenY+TIkXB1dUWZMmUQHh6OW7duFfm+ERERERERGciafK1evRqRkZGIjo7G0aNH0aBBAwQHB+P27dtm60+YMAHffvstvvrqK5w+fRojRoxA165dcezYMavaHDVqFDZu3Ig1a9Zg165duHHjBrp161bs+0tERERERC8uWZOv2bNnY9iwYRg8eDBq166NhQsXwsHBAXFxcWbrL1u2DOPGjUNISAiqVauGt99+GyEhIZg1a5bFbT58+BCxsbGYPXs22rZtC39/fyxevBj79u3DgQMHJNlvIiIiIiJ68djIteHMzEwcOXIEUVFRYplSqUS7du2wf/9+s+tkZGRAo9EYldnb2+OPP/6wuM0jR45Aq9WiXbt2Yp1atWqhcuXK2L9/P/73v//lue2MjAzxeUpKCoDsoZBardaaXaccDLFjDKXFuMuDcZcH4y4Pxl0ejLs8GHd5lKS4W9oH2ZKvu3fvQqfTwdPT06jc09MTZ8+eNbtOcHAwZs+ejZYtW8LPzw+JiYlYv349dDqdxW0mJyfDzs4OZcuWNamTnJycZ39jYmIwZcoUk/KtW7fCwcGhwP2l/CUkJMjdhRcS4y4Pxl0ejLs8GHd5MO7yYNzlURLinpaWZlE92ZKvwpg7dy6GDRuGWrVqQaFQwM/PD4MHD85zmGJRioqKQmRkpPg8JSUFPj4+aN++PZydnYt9+88rrVaLhIQEBAUFwdbWVu7uvDAYd3kw7vJg3OXBuMuDcZcH4y6PkhR3w6i4gsiWfLm5uUGlUpnMMnjr1i14eXmZXcfd3R0bNmxAeno67t27B29vb4wdOxbVqlWzuE0vLy9kZmbiwYMHRme/8tsuAKjVaqjVapNyW1tb2V/s5wHjKA/GXR6MuzwYd3kw7vJg3OXBuMujJMTd0u3LNuGGnZ0d/P39kZiYKJbp9XokJiaiadOm+a6r0WhQsWJFZGVlYd26dejSpYvFbfr7+8PW1taozrlz55CUlFTgdomIiIiIiApL1mGHkZGRGDhwIBo3boyAgADMmTMHqampGDx4MABgwIABqFixImJiYgAABw8exPXr19GwYUNcv34dkydPhl6vx8cff2xxmy4uLhg6dCgiIyNRvnx5ODs7491330XTpk3znGyDiIiIiIjoWcmafPXs2RN37tzBpEmTkJycjIYNG2LLli3ihBlJSUlQKv87OZeeno4JEybg8uXLKFOmDEJCQrBs2TKj4YMFtQkAX375JZRKJcLDw5GRkYHg4GB88803ku03ERERERG9eGSfcCMiIgIRERFml+3cudPoeatWrXD69OlnahPIHrY4f/58zJ8/36q+EhERERERFZasN1kmIiIiIiJ6UTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpIAky8iIiIiIiIJMPkiIiIiIiKSAJMvIiIiIiIiCTD5IiIiIiIikgCTLyIiIiIiIgkw+SIiIiIiIpKA7MnX/Pnz4evrC41Gg8DAQBw6dCjf+nPmzEHNmjVhb28PHx8fjBo1Cunp6eJyX19fKBQKk8fIkSPFOq1btzZZPmLEiGLbRyIiIiIiIhs5N7569WpERkZi4cKFCAwMxJw5cxAcHIxz587Bw8PDpH58fDzGjh2LuLg4NGvWDOfPn8egQYOgUCgwe/ZsAMDhw4eh0+nEdU6dOoWgoCB0797dqK1hw4Zh6tSp4nMHB4di2ksiIiIiIipKOr2Ag1fu48hdBVyv3EfT6h5QKRVyd6tAsiZfs2fPxrBhwzB48GAAwMKFC7Fp0ybExcVh7NixJvX37duHV199FX369AGQfZard+/eOHjwoFjH3d3daJ0ZM2bAz88PrVq1Mip3cHCAl5eXxX3NyMhARkaG+DwlJQUAoNVqodVqLW6HjBlixxhKi3GXB+MuD8ZdHoy7PBh3eTDu0vr9r1v4dPNZJKdkAFDhhwv/By9nNSaE1EJwHU9Z+mTpa68QBEEo5r6YlZmZCQcHB6xduxZhYWFi+cCBA/HgwQP8/PPPJuvEx8fjnXfewdatWxEQEIDLly+jU6dO6N+/P8aNG2d2G97e3oiMjDRa3rp1a/z1118QBAFeXl4IDQ3FxIkT8z37NXnyZEyZMsVsn3jWjIiI6PmgF4BLKQqkaAFnW8DPWUAp+DKd6IVx4p4CcecNV07l/HBmpzRDXtKjgav06U1aWhr69OmDhw8fwtnZOc96sp35unv3LnQ6HTw9jbNTT09PnD171uw6ffr0wd27d9G8eXMIgoCsrCyMGDHCbOIFABs2bMCDBw8waNAgk3aqVKkCb29vnDx5EmPGjMG5c+ewfv36PPsbFRWFyMhI8XlKSgp8fHzQvn37fANM+dNqtUhISEBQUBBsbW3l7s4Lg3GXB+MuD8ZdHqUx7r//dQsx4rfp2eT+Nt1apTHuzwPGXRo6vYCYWbsBZJhZqoACwG+3HPBx35aSD0E0jIoriKzDDq21c+dOTJ8+Hd988w0CAwNx8eJFvP/++/jkk08wceJEk/qxsbHo2LEjvL29jcqHDx8u/l6vXj1UqFABr732Gi5dugQ/Pz+z21ar1VCr1Sbltra2/JAVAcZRHoy7dHR6AUcNY9P/eVRqxqY/T/h+l0dpifuWUzfx7qoTyP19+a2UDLy76gQW9HsFHepWkKVvhVFa4l7SCIIAQcg+h6J/+rv+6SAxw+/C03p6AYAACBCQkSngsRZIydBDqdM9Lf+vDQGAXm/ajmE5kN2e8LQ9vT77p5CjzHj7T/uasz3BeJ2cfTVuI+e6pvube53svuQqy9nXp50w7GPOPpnGLWe5+Xb0ufv6NJh6QcD1f58YfTli8voBuPkwA8f+eYSmfq5F98awgKWfN9mSLzc3N6hUKty6dcuo/NatW3leizVx4kT0798fb775JoDsxCk1NRXDhw/H+PHjoVT+N3nj33//jW3btuV7NssgMDAQAHDx4sU8ky8iosLacuompmw8jZsP02EYm17BRYPo0Nql6mCOqKgYDrr0gmB0gGsoE/SA7ukyk+X6/57rBMG4Lb3xAbPRcr0gHnTm3LZeEJCVJSBq/Z8miRcAsWzsuj/xKD0LCoXC/MGseNBrekBc2INQk4P3p7/kPMj97+D96T7r9Lj6txL7fzkNhUIJ5DqQz30wK+TqF3L12/RA33BwbO6g3tzBe45ys8mCuYN60+3+14aZdvLbXo5tGL9O5hOaZ2OD8f+361kboSJw+1F6wZVkIlvyZWdnB39/fyQmJorXfOn1eiQmJiIiIsLsOmlpaUYJFgCoVCoA2R+qnBYvXgwPDw906tSpwL4cP34cAFChAg+CiKhobTl1E28vP2pyUJf8MB1vLz9a6r5Nl4IlB+aGA2tLDswzM7NwIxU4c/MRlCpVjvqC0Xp5HZgbDujNLReE7LOaZvuZ43edPv/lhu0bbydn3f+2U9By4/1DPv0wt0+GWDz7dvSCAK1WhTH/t81k+bMf5ErvwRMtRq89KXc3LKQEbv0jdydeWApF9pVISoXi6e9Pfz79XalA9q2ODHUVucsUZtrIUa54Wp6rrtiOuL3sOkqlaZlxG2bKzbWTo694uizn9pRPO5K73/+VG/bz6e9K03YM20DuNpTZfbjx8Ak2nrhZ4Gvg4aQpype0SMk67DAyMhIDBw5E48aNERAQgDlz5iA1NVWc/XDAgAGoWLEiYmJiAAChoaGYPXs2GjVqJA47nDhxIkJDQ8UkDMhO4hYvXoyBAwfCxsZ4Fy9duoT4+HiEhITA1dUVJ0+exKhRo9CyZUvUr19fup0nolJLEARk6QVkZumzHzo9MrR6ZOp0yDCUZenxJFNX4LfpH689iaT7aVBAkfeBuRUHxHktt/7Au3AH5pZsp6DlxXNgbgOc3F8cDVO+FIBe/0wtKJ8e1BkOxJRPD/4Mz1VKwzJFjrpPDxKV2c9VRuvm+P3p8odpWvx9P63AvtTycoKns6bgA2WjA9bCHYSKB7MwPVA2tJO7TIHsY6CLFy/gpZdego1KlffB7NPfTcqVuffLuK+FP6jP0U7uslzJiXECkFcykiuhURr3FbmSHKP9VZr21Tie//ULhtekgL5mabX47bff0KlTCId7FiOdXsD/Xf0XyQ/Tzf5vVQDwctEgoGp5qbtmMVmTr549e+LOnTuYNGkSkpOT0bBhQ2zZskWchCMpKcnoTNeECROgUCgwYcIEXL9+He7u7ggNDcW0adOM2t22bRuSkpIwZMgQk23a2dlh27ZtYqLn4+OD8PBwTJgwoXh3loieiV4vZCc5ORIeQ5KTkaX773edoey/5ZlZOpP1MrJyt6UzXs+ofUO5TuxDUSUIKelZmL7Z/CRDVLCCDsyVyuzn2sxM2Gs0hTowVygUUFm43CgBUBoO2LLLVHku/69tsf9Kc/uT3df8liufJiNmExELl2f3LVcccyzPuZ38luv1OuzeuRNt27aBna2tyfLc2zH3Giolui5y/6V76L3oQIH1okPrSH4dibW0Wi02Z5xHSBs/JgES0iv/S+6o+KiUCkSH1sbby49CARglYIbwR4fWLtHXVMs+4UZERESewwx37txp9NzGxgbR0dGIjo7Ot8327dubDEM08PHxwa5duwrVV6IXiaVnd0yTmhxJjNlE6GlbZpKd7PbNJFU6PbS6kjtOSaVUQG2jhJ2NEnaqpz9tlEjX6nHjwZMC1/evUhaVyzvmOui0/sDccICf33LTA978D7zNLVdYcOBd0HKlQvE0Cch7eUEH5ooc3+jnR6vVYvPmzQgJacWDUQlptVq4aoCKZe1LfNwDqpZHBRdNqf42nehF0aFuBSzo90qOa6mzeZWSa6llT77o2ej0Ag5duY/bj9Lh4ZT9j6EkZ/uUP2vO7uROVKw5u5Ou1SH5jgrfJx2AVieIdYrr7E5xsLNRQv000RETnxwJkNpGZVSmVimhtjVOjsQ6RmU52lOpTNtXGer8135enzlLv03/qH2tEv9tOtHz7Hn4Np3oRdKhbgUE1fbC/ou3sXXPQbRvEVhqZhFm8lWKGc+glo0zqFnHkrM7GbnO2DzL2Z18kyrJz+4ogEeW3ZMCyPvsjiEJUedKYPKsZ0hgciVC2WUqsUydR1tqGxVsVQqLznjIjd+mE5Uepf3bdKIXjUqpQGDV8rh3RkBgKTr5wOSrlCrNM6jlPLuTlp6B+xnA1Xup0EFplLw8y9kd8wkTz+4YymwUAk6dOI7/BTSGg9rumc7uUN74bTpR6WL4Np0jSoiouDD5KoV0egFTNp7OcwY1BYApG08jqLYXVEqFydmd/xKVZz+7k/E0iXm2szs2wNG9xR84C7woZ3e0Wi1U/xxDm5ruJf5ajNKO36YTlS4qpYLDgImo2DD5KoUOXblvdBCXmwDg5sN01IveAj1Q4s/u2CgEaOxsja+1KWHX7hA9i9I8Np2IiIiKDpOvUsjSu3anac3fW6Uknd2BPgu//fYbQkKCeQaGnmuldWw6ERERFR0mX6WQpXftntWjAQJ8y5fosztabcnpCxERERFRcWLyVQpZOoNaWMOKJSrRIiIiIiJ6kSnl7gBZzzCDGvDfjGkGnEGNiIiIiKhkYvJVShlmUPNyMR6C6OWiKdHTzBMRERERvag47LAU4/1IiIiIiIhKDyZfpRzvR0JEREREVDpw2CEREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBGRPvubPnw9fX19oNBoEBgbi0KFD+dafM2cOatasCXt7e/j4+GDUqFFIT08Xl0+ePBkKhcLoUatWLaM20tPTMXLkSLi6uqJMmTIIDw/H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b98GADg4OEChUMjcq4Lp9XpkZmYiPT2dU81LSKq4C4KAtLQ03L59G2XLls33XmsFeabky0ClUiEsLAxhYWFF0RwRERERvaC8vLJviGxIwEoDQRDw5MkT2Nvbl4pk8XkhddzLli0rvj8Lq0iSLyIiIiKioqBQKFChQgV4eHhAq9XK3R2LaLVa7N69Gy1btiyVN7curaSMu62t7TOd8TJg8kVEREREJY5KpSqSg10pqFQqZGVlQaPRMPmSUGmMOwelEhERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSUD25Gv+/Pnw9fWFRqNBYGAgDh06lG/9OXPmoGbNmrC3t4ePjw9GjRqF9PR0q9pMT0/HyJEj4erqijJlyiA8PBy3bt0q8n0jIiIiIiIykDX5Wr16NSIjIxEdHY2jR4+iQYMGCA4Oxu3bt83Wj4+Px9ixYxEdHY0zZ84gNjYWq1evxrhx46xqc9SoUdi4cSPWrFmDXbt24caNG+jWrVux7y8REREREb24ZE2+Zs+ejWHDhmHw4MGoXbs2Fi5cCAcHB8TFxZmtv2/fPrz66qvo06cPfH190b59e/Tu3dvozFZBbT58+BCxsbGYPXs22rZtC39/fyxevBj79u3DgQMHJNlvIiIiIiJ68djIteHMzEwcOXIEUVFRYplSqUS7du2wf/9+s+s0a9YMy5cvx6FDhxAQEIDLly9j8+bN6N+/v8VtHjlyBFqtFu3atRPr1KpVC5UrV8b+/fvxv//9z+y2MzIykJGRIT5PSUkBAGi1Wmi12kJGgQyxYwylxbjLg3GXB+MuD8ZdHoy7PBh3eZSkuFvaB9mSr7t370Kn08HT09Oo3NPTE2fPnjW7Tp8+fXD37l00b94cgiAgKysLI0aMEIcdWtJmcnIy7OzsULZsWZM6ycnJefY3JiYGU6ZMMSnfunUrHBwcCtxfyl9CQoLcXXghMe7yYNzlwbjLg3GXB+MuD8ZdHiUh7mlpaRbVky35KoydO3di+vTp+OabbxAYGIiLFy/i/fffxyeffIKJEycW67ajoqIQGRkpPk9JSYGPjw/at28PZ2fnYt3280yr1SIhIQFBQUGwtbWVuzsvDMZdHoy7PBh3eTDu8mDc5cG4y6Mkxd0wKq4gsiVfbm5uUKlUJrMM3rp1C15eXmbXmThxIvr3748333wTAFCvXj2kpqZi+PDhGD9+vEVtenl5ITMzEw8ePDA6+5XfdgFArVZDrVablNva2sr+Yj8PGEd5MO7yYNzlwbjLg3GXB+MuD8ZdHiUh7pZuX7YJN+zs7ODv74/ExESxTK/XIzExEU2bNjW7TlpaGpRK4y6rVCoAgCAIFrXp7+8PW1tbozrnzp1DUlJSntslIiIiIiJ6VrIOO4yMjMTAgQPRuHFjBAQEYM6cOUhNTcXgwYMBAAMGDEDFihURExMDAAgNDcXs2bPRqFEjcdjhxIkTERoaKiZhBbXp4uKCoUOHIjIyEuXLl4ezszPeffddNG3aNM/JNoiIiIiIiJ6VrMlXz549cefOHUyaNAnJyclo2LAhtmzZIk6YkZSUZHSma8KECVAoFJgwYQKuX78Od3d3hIaGYtq0aRa3CQBffvkllEolwsPDkZGRgeDgYHzzzTfS7TgREREREb1wZJ9wIyIiAhEREWaX7dy50+i5jY0NoqOjER0dXeg2AUCj0WD+/PmYP3++1f0lIiIiIiIqDFlvskxERERERPSiYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfREREREREEmDyRUREREREJAEmX0RERERERBJg8kVERERERCQBJl9EREREREQSYPJFREREREQkASZfRERE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HRIREREREb0ImHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMkXERERERGRBJh8ERERERERSYDJFxERERERkQSYfBEREREREUmAyRcREREREZEEmHwRERERERFJgMnXcyA5NRmHbh5Ccmqy3F0hIiIiIqI82MjdAXo26y+sx5T9U6AX9FAqlIhuGo1uNbrJ3S0iIiIiIsqFyVcplpyajCn7pkAPPQBAL+gRvS8av13+Dc5qZ6hVatip7KBWqY1+L2yZndIOKqVK5r0mIiIiIiqdmHyVYkkpSWLildOB5APFtk0bpY35JE1ZuKROBRX+yvwLzted4aB2YPJHRERERM8tJl+lWGXnylBCaZSAKaDAKP9R0NhokKnLRIYuAxm6DPF3a8sydZnIErLE9rP0WcjSZyFVm1qk+7Jy10qL6hV18sczf0REREQkFSZfpZiXoxeim0UX+zVfWfosZOoyTZM1feGTOsPv6VnpSL6TDEcXR2TqzWxDlwGdoDPqS3Ekf5Zi8kdEREREhcXkq5TrVqMbmnk3w7VH1+Dj5AMvR68i34aN0gY2Shs42DoUedtarRabN29GSIcQ2Nramq1jSP4sSe4KTP701iWLGboM6AW9UV+eh+TPBjY4l3EOiqsKONg5WLyuUsEJUomIiKhkuJV2C5e1l3Er7RYquVSSuzsWYfL1HPBy9CqWpKukKM7kzxLPc/L3076frKov55k/Jn9EJJXk1GQkpSShsnPl5/r/K1Fptv7CenHiuSUbliC6WemY8ZvJF1EBnsfkL12bjn9u/gMXVxdoBe0Ld+ZPruSvNH5DR/Q8EQQh+ycE8bn4+9OfGy5swKcHPoUeeiihxLjAcehSvYvROmJ7ebSTs05+yw1lBfXN7PLc7TytmpmViXu6e0hKSYLKRmWyjqGeuX7k7nvufRaQo095tGNxnAqIgdk45NxmHnHKbzs528n9upu8F3Jsy+zyXPuYlZWF45nHobusg1KlNBuDguJtbjsFxsmS95KV7RQUb3N9tzjeRfReSc1Mxa9XfhWf66HHlP1T0My7WYn/wkQh5H7VySIpKSlw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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}
