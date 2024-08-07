import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

class DecisionTree:

    def __init__(self, trainData: pd.DataFrame):
        self.trainData = trainData
        self.X_train = self.trainData.drop(columns=['Survived'], axis=1)
        self.Y_train = self.trainData['Survived']
    
    def train(self, criteriation=None, max_depth=None, min_sample_splt=None, min_sample_leaf=None):
        """
        This method trains a Decision Tree model on the data provided.
        """
        if criteriation is None:
            model = DecisionTreeClassifier()
        else:
            model = DecisionTreeClassifier(criterion=criteriation, max_depth=max_depth, min_samples_split=min_sample_splt, min_samples_leaf=min_sample_leaf)

        self.model = model.fit(self.X_train, self.Y_train)

        tp, fp, fn, tn = confusion_matrix(self.Y_train, model.predict(self.X_train)).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) 
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"Model trained with accuracy: {accuracy}, recall: {recall}, precision: {precision}, specificity: {specificity}, f1_score: {f1}")
        
        if criteriation is not None:            
            return [accuracy, recall, precision, specificity, f1]

    def tunnig(self):
        """
        This method performs a pseudo grid search to find the best hyperparameters for the Decision Tree model.
        """
        criterionList = ['gini', 'entropy']
        max_depthList = range(1, 10)
        min_samples_splitList = range(1, 10)
        min_samples_leafList = range(1, 10)

        best_acurracy = 0.0

        for criterion in criterionList:
            for max_depth in max_depthList:
                for min_samples_split in min_samples_splitList:
                    for min_samples_leaf in min_samples_leafList:
                        acurracy, recall, precision, specificity, f1 = self.train(criterion, max_depth, min_samples_split, min_samples_leaf)
                        if acurracy > best_acurracy:
                            best_acurracy = acurracy
                            best_params = [criterion, max_depth, min_samples_split, min_samples_leaf]
        
        print(f"Best params: {best_params} with acurracy: {best_acurracy}")
        self.model = DecisionTreeClassifier(criterion=best_params[0], max_depth=best_params[1], min_samples_split=best_params[2], min_samples_leaf=best_params[3])

    def predict(self, testData: pd.DataFrame):
        """
        This method predicts the output of the data provided.
        """
        try:
            self.model.predict(testData)
            # provavelmente iremos fazer mais alguma coisa que deve ser colocada aqui            
        except:
            raise Exception("Model not trained yet. Please train the model first.")
