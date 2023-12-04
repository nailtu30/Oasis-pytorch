import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances


class ISODATA():
    def __init__(self, data, designCenterNum, LeastSampleNum, StdThred, LeastCenterDist, iterationNum):
        #  指定预期的聚类数、每类的最小样本数、标准差阈值、最小中心距离、迭代次数
        self.K = designCenterNum
        self.thetaN = LeastSampleNum
        self.thetaS = StdThred
        self.thetaC = LeastCenterDist
        self.iteration = iterationNum

        # # 初始化
        # self.n_samples = 1500
        # # 选一
        # self.random_state1 = 200
        # self.random_state2 = 160
        # self.random_state3 = 170
        # self.data, self.label = make_blobs(
        #     n_samples=self.n_samples, n_features=2, random_state=self.random_state3)

        self.data = data

        self.center = self.data[0, :].reshape((1, -1))
        self.centerNum = 1
        self.centerMeanDist = 0

        # seaborn风格
        # sns.set()

    def updateLabel(self):
        """
            更新中心
        """
        for i in range(self.centerNum):
            # 计算样本到中心的距离
            if np.any(np.isnan(self.center)):
                print('NAN!!!!')

            distance = euclidean_distances(
                self.data, self.center.reshape((self.centerNum, -1)))
            print('distance shape: {}'.format(distance.shape))
            print(self.center[i, :])
            # 为样本重新分配标签
            self.label = np.argmin(distance, 1)
            print('i : {}'.format(i))
            print('label: {}'.format(self.label))
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            print('index: {}'.format(index))
            # if index.size == 0:
            #     continue
            sameClassSample = self.data[index, :]
            if np.any(np.isnan(sameClassSample)):
                print('same class NAN!!!')
            # 更新中心
            res = np.mean(sameClassSample, 0)
            res = np.nan_to_num(res, nan=0)

            # res = np.nanmean(sameClassSample, 0)
            print('sameClass shape: {}'.format(sameClassSample.shape))
            # print('res shape: {}'.format(res.shape))
            print(sameClassSample)
            if np.any(np.isnan(res)):
                print('res NAN!!!')
            self.center[i, :] = res
            if np.any(np.isnan(self.center)):
                print('mean NAN!!!!')

        # 计算所有类到各自中心的平均距离之和
        for i in range(self.centerNum):
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            # 计算样本到中心的距离
            distance = np.mean(euclidean_distances(
                sameClassSample, self.center[i, :].reshape((1, -1))))
            # 更新中心
            self.centerMeanDist += distance
        self.centerMeanDist /= self.centerNum

    def divide(self):
        # 临时保存更新后的中心集合,否则在删除和添加的过程中顺序会乱
        newCenterSet = self.center
        # 计算每个类的样本在每个维度的标准差
        for i in range(self.centerNum):
            # 找出同一类样本
            index = np.argwhere(self.label == i).squeeze()
            sameClassSample = self.data[index, :]
            # 计算样本到中心每个维度的标准差
            stdEachDim = np.mean(
                (sameClassSample - self.center[i, :])**2, axis=0)
            # 找出其中维度的最大标准差
            maxIndex = np.argmax(stdEachDim)
            maxStd = stdEachDim[maxIndex]
            # 计算样本到本类中心的距离
            distance = np.mean(euclidean_distances(
                sameClassSample, self.center[i, :].reshape((1, -1))))
            # 如果最大标准差超过了阈值
            if maxStd > self.thetaS:
                # 还需要该类的样本数大于于阈值很多 且 太分散才进行分裂
                if self.centerNum <= self.K//2 or \
                        sameClassSample.shape[0] > 2 * (self.thetaN+1) and distance >= self.centerMeanDist:
                    newCenterFirst = self.center[i, :].copy()
                    newCenterSecond = self.center[i, :].copy()

                    newCenterFirst[maxIndex] += 0.5 * maxStd
                    newCenterSecond[maxIndex] -= 0.5 * maxStd

                    # 删除原始中心
                    newCenterSet = np.delete(newCenterSet, i, axis=0)
                    # 添加新中心
                    newCenterSet = np.vstack((newCenterSet, newCenterFirst))
                    newCenterSet = np.vstack((newCenterSet, newCenterSecond))
                    print('divide su')
                else:
                    print('divide un')

            else:
                print('maxSTD: {}'.format(maxStd))
                print('divide un thetaS')
                continue
        # 更新中心集合
        self.center = newCenterSet
        self.centerNum = self.center.shape[0]
        if np.any(np.isnan(self.center)):
            print('divide NAN!!!!')

    def combine(self):
        # 临时保存更新后的中心集合,否则在删除和添加的过程中顺序会乱
        delIndexList = []

        # 计算中心之间的距离
        centerDist = euclidean_distances(self.center, self.center)
        centerDist += (np.eye(self.centerNum)) * 10**10
        # 把中心距离小于阈值的中心对找出来
        while True:
            # 如果最小的中心距离都大于阈值的话，则不再进行合并
            minDist = np.min(centerDist)
            if minDist >= self.thetaC:
                break
            # 否则合并
            index = np.argmin(centerDist)
            row = index // self.centerNum
            col = index % self.centerNum
            # 找出合并的两个类别
            index = np.argwhere(self.label == row).squeeze()
            classNumFirst = len(index)
            index = np.argwhere(self.label == col).squeeze()
            classNumSecond = len(index)
            newCenter = self.center[row, :] * (classNumFirst / (classNumFirst + classNumSecond)) + \
                self.center[col, :] * \
                (classNumSecond / (classNumFirst + classNumSecond))
            # 记录被合并的中心
            delIndexList.append(row)
            delIndexList.append(col)
            # 增加合并后的中心
            self.center = np.vstack((self.center, newCenter))
            self.centerNum -= 1
            # 标记，以防下次选中
            centerDist[row, :] = float("inf")
            centerDist[col, :] = float("inf")
            centerDist[:, col] = float("inf")
            centerDist[:, row] = float("inf")

        # 更新中心
        self.center = np.delete(self.center, delIndexList, axis=0)
        self.centerNum = self.center.shape[0]
        if np.any(np.isnan(self.center)):
            print('combine NAN!!!!')

    # def drawResult(self, iteration):
    #     ax = plt.gca()
    #     ax.clear()
    #     ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label, cmap="cool")
    #     # ax.set_aspect(1)
    #     # 坐标信息
    #     ax.set_xlabel('x axis')
    #     ax.set_ylabel('y axis')
    #     # plt.show()
    #     plt.savefig('isofigures/{}.png'.format(iteration))

    def train(self):
        # 初始化中心和label
        self.updateLabel()
        # self.drawResult(-1)

        # 到设定的次数自动退出
        for i in range(self.iteration):
            # 如果是偶数次迭代或者中心的数量太多，那么进行合并
            if self.centerNum < self.K // 2:
                self.divide()
                if np.any(np.isnan(self.center)):
                    print('train divide NAN!!!!')
            elif (i > 0 and i % 2 == 0) or self.centerNum > 2 * self.K:
                self.combine()
                if np.any(np.isnan(self.center)):
                    print('train combine NAN!!!!')
            else:
                self.divide()
                if np.any(np.isnan(self.center)):
                    print('train divide NAN!!!!2')
            # 更新中心
            self.updateLabel()
            if np.any(np.isnan(self.center)):
                print('train update label NAN!!!!')
            # self.drawResult(i)
            print("中心数量：{}".format(self.centerNum))

        print('center: {}'.format(self.center))
        print('label: {}'.format(self.label))


if __name__ == "__main__":
    isoData = ISODATA(designCenterNum=5, LeastSampleNum=20,
                      StdThred=0.1, LeastCenterDist=5, iterationNum=20)
    isoData.train()
