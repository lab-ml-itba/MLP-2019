from matplotlib import pyplot as plt
import keras
import numpy as np
from IPython.display import clear_output
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures

class PlotCallback(keras.callbacks.Callback):
    def get_loss_by_sample(self, y_true, y_pred, eps=1e-15):
        # y_pred = np.clip(y_pred, eps, 1 - eps)
        # print(y_pred)
        losses = - y_true * np.log(y_pred + eps) - (1-y_true) * np.log(1-y_pred + eps)
        idxs = np.where(np.isnan(losses))[0]
        if len(idxs)>0:
            print(y_pred[idxs[0]])
        # print()
        return losses
    
    def plotBoundary(self, model=None):
        if model is None:
            model = self.model
        eps=1e-15
        clear_output(wait=True)
        fig=plt.figure(figsize=(20,11))
        gs=GridSpec(3, 3) # 2 rows, 3 columns
        
        ax=fig.add_subplot(gs[:2,:2]) # Second row, span all columns
        axLoss=fig.add_subplot(gs[0,2]) # First row, first column
        axAcc=fig.add_subplot(gs[1,2]) # First row, second column
        axLossHist = fig.add_subplot(gs[2,1])
        axLogOddsHist = fig.add_subplot(gs[2,0])
        predictions = model.predict_proba(self.data_poly).reshape(-1)
        predictions_0 = predictions[self.class_0]
        predictions_1 = predictions[self.class_1]
        log_odds_0 = np.log(eps + predictions_0/(1 - predictions_0 + eps))
        log_odds_1 = np.log(eps + predictions_1/(1 - predictions_1 + eps))
        axLogOddsHist.hist(log_odds_0, self.bins, color='r')
        axLogOddsHist.hist(log_odds_1, self.bins, color='b', alpha=0.5)
        
        losses_0 = self.get_loss_by_sample(np.zeros(len(predictions_0)), predictions_0)
        losses_1 = self.get_loss_by_sample(np.ones(len(predictions_1)), predictions_1)
        
        axLossHist.hist(losses_0, self.bins, color='r')
        axLossHist.hist(losses_1, self.bins, color='b', alpha=0.5)
        
        axROC = fig.add_subplot(gs[2,2])
        auROC = roc_auc_score(self.labels, predictions)
        axROC.set_title(f'ROC curve - AuROC:{auROC:.4f}')
        fpr, tpr, thres = roc_curve(self.labels, predictions)
        axROC.plot(fpr, tpr)
        
        ax.scatter(self.data[self.class_1][:,0], self.data[self.class_1][:,1], color='b', s=5, alpha=0.5)
        ax.scatter(self.data[self.class_0][:,0], self.data[self.class_0][:,1], color='r', s=5, alpha=0.5)
        Z = 1 - model.predict_proba(self.grid_data_poly)[:, 0]
        Z = Z.reshape(self.Z_shape)
        ax.contour(self.X, self.Y, Z, (0.5,), colors='k', linewidths=0.5)
        axAcc.plot(self.acc)
        if len(self.acc)==0:
            loss, acc = model.evaluate(self.data_poly, self.labels, verbose=0)
            axAcc.set_title(f'Accuracy: {acc:.4f}')
            axLoss.set_title(f'Cross Entropy: {loss:.4f}')
        else:
            axAcc.set_title(f'Accuracy: {self.acc[-1]:.4f}')
            axLoss.set_title(f'Cross Entropy: {self.loss[-1]:.4f}')
        
        axLoss.plot(self.loss)
            
        axLossHist.set_title('Cross Entropy Histogram')
        axLogOddsHist.set_title('Log odds Histogram')
        plt.show()
        
        
    def __init__(self, data, labels, plots_every_batches=100, N = 300, bins=100, degree=1):

        polyFeat = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        self.data_poly = polyFeat.fit_transform(data)
        
        self.plots_every_batches = plots_every_batches
        self.bins = bins
        self.N = N
        self.data = data
        self.labels = labels
        mins = data[:,:2].min(axis=0)
        maxs = data[:,:2].max(axis=0)
        X_lin = np.linspace(mins[0], maxs[0], self.N)
        Y_lin = np.linspace(mins[1], maxs[1], self.N)
        self.X, self.Y = np.meshgrid(X_lin, Y_lin)
        self.Z_shape = self.X.shape
        self.grid_data = np.c_[self.X.flatten(), self.Y.flatten()]
        self.grid_data_poly = polyFeat.transform(self.grid_data)
        self.acc = []
        self.loss = []
        self.class_1 = labels == 1
        self.class_0 = labels == 0
        
    def on_train_begin(self, logs={}):
        self.plotBoundary()
        return
    
    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.plotBoundary()
        return
    
    def on_batch_end(self, batch, logs={}):
        # if batch%self.plots_every_batches == 0:
        #    self.acc.append(logs.get('acc'))
        #    self.loss.append(logs.get('loss'))
        #    self.plotBoundary()
        return