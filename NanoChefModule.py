import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time
import pickle
import torch
from torchinfo import summary
from collections import Counter
from pyDOE import lhs
from itertools import combinations, permutations
import warnings
import dill
warnings.filterwarnings('ignore')

# from olympus.noises.noise_gaussian_noise import GaussianNoise
from Sequence.SeqOpt import SeqOpt
from Sequence.acq.acq_func_BNN_seq import acq_max_real
from Log.Logging_Class import ModelLogger
from Sequence.utils.functions import caculateMAE_seq_real, load_json_to_dict, train_seq_real, output_space_property
from Sequence.utils.target_space import SeqDiscreteSpace

np.set_printoptions(suppress=True,threshold=np.inf, precision=6)


class NanoChef():
    
    def __init__(self, algorithm_dict:dict):
        """
        Definition of variables
        """
        # self.subject=algorithm_dict["subject"]
        # self.group=algorithm_dict["group"]
        # self.logLevel=algorithm_dict["logLevel"]

        # self.model=algorithm_dict["model"]
        # self.batchSize=algorithm_dict["batchSize"]
        # self.totalCycleNum=algorithm_dict["totalCycleNum"]

        # self.samplingMethod = self.sampling["samplingMethod"]
        # self.samplingNum = self.sampling["samplingNum"] # sampling의 개수

        # self.reagent_list=self.structure["reagent_list"]
        # self.ps_dim=self.structure["ps_dim"] # the dimension of positional encoding
        # self.output_dim=self.structure["output_dim"] # hyperparameter of positional encoding
        # self.nn_n_hidden=self.structure["nn_n_hidden"] # hyperparameter of hidden layer
        # self.randomState=self.structure["randomState"]
        # self.n_train_epochs=self.structure["n_train_epochs"]
        # self.lr=self.structure["lr"]
        # self.patience=self.structure["patience"]

        # self.acqMethod = self.acq["acqMethod"]
        # self.acqSampler = self.acq["acqSampler"]
        # self.acqHyperparameter = self.acq["acqHyperparameter"] # kappa (exploration)
        # self.acq_n_samples = self.acq["acq_n_samples"]
        
        # self.lossMethod = self.loss["lossMethod"]
        # self.lossTarget = self.loss["lossTarget"]

        # self.prange=algorithm_dict["prange"]
        self.algorithm_dict=algorithm_dict

        for key, value in algorithm_dict.items():
            setattr(self, key, value)
        for key, value in self.sampling.items():
            setattr(self, key, value)
        for key, value in self.structure.items():
            setattr(self, key, value)
        for key, value in self.acq.items():
            setattr(self, key, value)
        for key, value in self.loss.items():
            setattr(self, key, value)

        self.normPrange=self._getNormalizeList(self.prange)
        self._norm_space=SeqDiscreteSpace(prange=self.normPrange, target_func=None, target_condition_dict=self.lossTarget, random_state=self.randomState)
        self._real_space=SeqDiscreteSpace(prange=self.prange, target_func=None, target_condition_dict=self.lossTarget, random_state=self.randomState)

        BASIC_PATH="{}".format("Data")
        dir_name=time.strftime("%Y%m%d")
        self.TOTAL_LOG_FOLDER = "{}/{}/{}/{}/{}".format(BASIC_PATH, self.subject, "Log", self.modeType, dir_name)
        self.TOTAL_MODEL_FOLDER = "{}/{}/{}/{}/{}".format(BASIC_PATH, self.subject, "Model", self.modeType, dir_name)
        self.TOTAL_OBJECT_FOLDER = "{}/{}/{}/{}/{}".format(BASIC_PATH, self.subject, "Object", self.modeType, dir_name)
        self.TOTAL_RESULT_FOLDER = "{}/{}/{}/{}/{}".format(BASIC_PATH, self.subject, "Result", self.modeType, dir_name)
        if os.path.isdir(self.TOTAL_LOG_FOLDER) == False:
            os.makedirs(self.TOTAL_LOG_FOLDER)
            os.makedirs(self.TOTAL_MODEL_FOLDER)
            os.makedirs(self.TOTAL_OBJECT_FOLDER)
            os.makedirs(self.TOTAL_RESULT_FOLDER)
        
        self.model_logger=ModelLogger(self.subject, self.logLevel, self.TOTAL_LOG_FOLDER)

        np.random.seed(self.randomState)
        self.SeqOpt_obj=SeqOpt(
            reagent_list=self.reagent_list,
            ps_dim=self.ps_dim,
            output_dim=self.output_dim, # output dimension of positional encoding
            num_variables=len(self.prange), # output dimension of conditional vector
            nn_n_hidden=self.nn_n_hidden, # hyperparameter
            seed_num=self.randomState,
            device=self.device).to(self.device)
        self.model_logger.info("SeqOpt ({})".format("model summary"), "{}".format(summary(self.SeqOpt_obj)))
        
        self.optimizer = torch.optim.Adam(self.SeqOpt_obj.parameters(), lr=self.lr)
        
        self.best_loss_list=[]
        self.best_mae_list=[]
        self.best_y_list=[]
        
        self.countSamplingNum=0

        self.reagent_seq_tuple=list(permutations(self.reagent_list)) # Definition of reagent sequences. e.g.) reagent_list=["A","B","C"]
        """
        reagent_seq_tuple = [
            ('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), 
            ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')
        ]
        """
        # self.reagent_seqs = np.array([np.array(t) for t in self.reagent_seq_tuple])
        self.reagent_seqs = [t for t in self.reagent_seq_tuple]
        """
        reagent_seqs = [
            ['A', 'B', 'C'],
            ['A', 'C', 'B'],
            ['B', 'A', 'C'],
            ['B', 'C', 'A'],
            ['C', 'A', 'B'],
            ['C', 'B', 'A']
        ]
        """
        self.sampling_cond_list=self._generateCondSampling(self.samplingMethod, self.samplingNum)
        self.sampling_seq_list=self._generateSeqSampling(self.reagent_seqs, self.samplingNum)

    def _getNormalizeList(self, prange):
        '''
        :param prange (dict) : 
        
        ex)
            {
                "AgNO3" : [100, 3000, 50],
                "H2O2" : [100, 3000, 50],
                "NaBH4": [100, 3000, 50]
            }

        :return : normPrange
        '''
        normPrange={}
        for chemical, rangeList in prange.items():
            new_range_list=[]
            
            new_range_list.append(0) # normalize min value = 0
            new_range_list.append(1) # normalize max value = 1
            new_range_list.append(rangeList[2]/(rangeList[1]-rangeList[0]))

            normPrange[chemical] = new_range_list
        return normPrange

    def _getNormalizedCondition(self, real_next_points):
        """
        convert real condition to normalized condition
        X' = (value - V_min)/(V_max - V_min) 

        :param real_next_points (list) : 
            [
                {'AgNO3': 3300.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3100.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 800.0,  'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 3500.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
            ]
        :return : normalized_next_points (list)
        """
        normalized_next_points = []
        for _, next_point in enumerate(real_next_points):
            new_value={}
            for chemical, rangeList in self.prange.items():
                new_value[chemical]=(int(next_point[chemical])-rangeList[0])/(rangeList[1]-rangeList[0]) # X' = (value - V_min)/(V_max - V_min) 
            normalized_next_points.append(new_value)
        
        return normalized_next_points

    def _getRealCondition(self, normalized_next_points):
        """
        convert normalized condition to real condition

        :param normalized_next_points (list) : 
        ex) [
                {'AgNO3': 0.02123154896, 'Citrate': 0.4563211120887, 'H2O': 0.122471125, 'H2O2': 0.6337412354, 'NaBH4': 1.0}
                ...
            ]
        :return : real_next_points (list)
        ex) [
                {'AgNO3': 3300.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3100.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 800.0,  'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 3500.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
            ]
        """
        real_next_points = []
        for _, normalized_next_point in enumerate(normalized_next_points):
            new_value={}
            for chemical, rangeList in self.prange.items():
                new_value["{}".format(chemical)]=round(normalized_next_point[chemical]*(rangeList[1]-rangeList[0])+rangeList[0])
            real_next_points.append(new_value)
            
        return real_next_points

    def _register(self, space, seq:list, params:dict, propertys:dict, target:float):
        """
        space.register(params, target)
        """
        space.register(seq, params, propertys, target)

    def _generateCondSampling(self,sampling_method:str,experiment_num:int):
        """
        :param sampling_method (str) : grid or random or latin
        :param experiment_num (int) : the number of experiments

        :return sampling_list (dicts in list)
        [
            {'AddSolution=AgNO3_Concentration': 125, 'AddSolution=AgNO3_Volume': 150, 'AddSolution=AgNO3_Injectionrate': 150}, 
            {'AddSolution=AgNO3_Concentration': 25, 'AddSolution=AgNO3_Volume': 750, 'AddSolution=AgNO3_Injectionrate': 50}, 
            {'AddSolution=AgNO3_Concentration': 250, 'AddSolution=AgNO3_Volume': 550, 'AddSolution=AgNO3_Injectionrate': 50}, 
            {'AddSolution=AgNO3_Concentration': 300, 'AddSolution=AgNO3_Volume': 350, 'AddSolution=AgNO3_Injectionrate': 50}, 
            {'AddSolution=AgNO3_Concentration': 325, 'AddSolution=AgNO3_Volume': 450, 'AddSolution=AgNO3_Injectionrate': 100}, 
            {'AddSolution=AgNO3_Concentration': 75, 'AddSolution=AgNO3_Volume': 950, 'AddSolution=AgNO3_Injectionrate': 100}, 
            {'AddSolution=AgNO3_Concentration': 175, 'AddSolution=AgNO3_Volume': 800, 'AddSolution=AgNO3_Injectionrate': 150}, 
            {'AddSolution=AgNO3_Concentration': 200, 'AddSolution=AgNO3_Volume': 1150, 'AddSolution=AgNO3_Injectionrate': 150}
        ]
        """
        # sampling_name="{}_sample"
        # if self.constraints == False:
        #     sample=getattr(self._norm_space, sampling_name)(experiment_num).tolist()
        #     sampling_cond_list = [] 
        #     for i in range(len(sample)):
        #         round_sample= self._norm_space._bin(sample[i])
        #         sampling_cond_list.append(self._norm_space.array_to_params(round_sample))
        # else:
        #     sampling_name="{}_sample_constraints"
        #     sample=getattr(self._norm_space, sampling_name)(experiment_num, self.constraints).tolist()
        #     sampling_cond_list = [] 
        #     for i in range(len(sample)):
        #         round_sample= self._norm_space._bin(sample[i])
        #         sampling_cond_list.append(self._norm_space.array_to_params(round_sample))
        
        if sampling_method == "latin":
            sample =self._norm_space.latin_sample(experiment_num).tolist()        
            sampling_cond_list = [] 
            for i in range(len(sample)):
                round_sample= self._norm_space._bin(sample[i])
                sampling_cond_list.append(self._norm_space.array_to_params(round_sample))
        elif sampling_method == "sobol":
            sample =self._norm_space.sobol_sample(experiment_num).tolist()        
            sampling_cond_list = [] 
            for i in range(len(sample)):
                round_sample= self._norm_space._bin(sample[i])
                sampling_cond_list.append(self._norm_space.array_to_params(round_sample))
        elif sampling_method == "random":          
            sampling_cond_list = [self._norm_space.array_to_params(self.space._bin(
                        self._norm_space.random_sample(constraints=self.get_constraint_dict()))) for _ in range(self.batchSize)]

        return sampling_cond_list
    
    def _generateSeqSampling(self,reagent_seqs:list,experiment_num:int):
        """
        :param experiment_num (int) : the number of experiments

        :return sampling_list (dicts in list)
        [
        ]
        """
        reagent_len=len(reagent_seqs)

        quotient=experiment_num//reagent_len
        remainder=experiment_num%reagent_len

        index_list=[]
        for idx in range(reagent_len):
            if idx < remainder:
                index_list.append(quotient+1)
            else:
                index_list.append(quotient)

        sampling_seq_list=[]
        for idx, counts in enumerate(index_list):
            for _ in range(counts):
                sampling_seq_list.append(reagent_seqs[idx])

        return sampling_seq_list
    
    # def _pop_n_items(self, lst:list, n:int):
    #     popped_items = []
    #     for _ in range(n):
    #         if lst:
    #             popped_items.append(lst.pop(0))
    #         else:
    #             break
    #     return popped_items
    
    def _pop_n_items(self, lst:list, sampling_num:int, n:int):
        return lst[sampling_num:sampling_num+n]
    
    def _suggestAI(self, experiment_num:int):
        if type(self.acqMethod) == str:
            print('self.acqHyperparameter["kappa"]', self.acqHyperparameter["kappa"])
            norm_seq_points, norm_next_points=acq_max_real(
                        acq_method=self.acqMethod,
                        seq_opt_obj=self.SeqOpt_obj,
                        model_logger=self.model_logger,
                        space=self._norm_space,
                        reagent_seqs=self.reagent_seqs,
                        acq_n_samples=self.acq_n_samples,
                        batch_size=self.batchSize,
                        kappa=self.acqHyperparameter["kappa"],
                        # device=self.device
                        )
            if len(norm_next_points) > experiment_num:
                norm_seq_points, trash_list = self._checkCandidateNumber(experiment_num, norm_seq_points)
                norm_next_points, trash_list = self._checkCandidateNumber(experiment_num, norm_next_points)
                # print("preprcoess norm_next_points: ",len(norm_next_points))
            elif len(norm_next_points)<experiment_num:
                raise ValueError("Candiate of condition is not match to batch_size. norm_next_points: {} != batch_size: {}. Please check this part.".format(str(len(norm_next_points)), str(experiment_num)))
        # list인 경우는 각각 batch의 갯수를 count한 후, 해당 function으로 해당 경우의 수만 suggest 진행
        elif type(self.acqMethod) == list:
            norm_next_points=[]
            # if acqMethod == ['ucb', 'ucb', 'ucb', 'ucb', 'ei', 'ei', 'es', 'es']
            # ucb 4개, ei 2개, es 2개 의 경우를 suggest 한다
            if len(self.acqMethod) == experiment_num:
                acqMethod_dict = Counter(self.acqMethod)
                # acqMethod_dict = {'ucb':4,'ei':2,'es':2}
                for acq_func_str, count in acqMethod_dict.items():
                    norm_seq_points, norm_next_points=acq_max_real(
                        acq_method=acq_func_str,
                        seq_opt_obj=self.SeqOpt_obj,
                        model_logger=self.model_logger,
                        space=self._norm_space,
                        reagent_seqs=self.reagent_seqs,
                        n_acq_samples=self.n_acq_samples,
                        batch_size=count,
                        kappa=self.acqHyperparameter["kappa"],
                        device=self.device
                        )
                    norm_seq_points, trash_list = self._checkCandidateNumber(experiment_num, norm_seq_points)
                    norm_next_points, trash_list = self._checkCandidateNumber(experiment_num, norm_next_points)
            else:
                raise IndexError("Please fill utility list to match experiment_num")
        else:
            raise TypeError("Please give string type or filled list")
        
        return norm_seq_points, norm_next_points

    def suggestNextStep(self):
        """
        recommend next synthesis recipe with sampling method or acquisition function

        :return next_points (dict): candidate of condition
        """
        # init random이 suggest_num보다 작을 때 초기 sampling \
        self.countSamplingNum=len(self._norm_space.res())
        print("self.countSamplingNum", self.countSamplingNum)
        print("self.samplingNum", self.samplingNum)
        print("self.countSamplingNum+self.batchSize", self.countSamplingNum+self.batchSize)

        if self.countSamplingNum < self.samplingNum and (self.countSamplingNum+self.batchSize) <= self.samplingNum:
            self.model_logger.info("SeqOpt ({})".format("suggestNextStep-sampling"), "initial sampling")
            real_seq_points=self._pop_n_items(self.sampling_seq_list, self.countSamplingNum, self.batchSize)
            norm_cond_points=self._pop_n_items(self.sampling_cond_list, self.countSamplingNum, self.batchSize)
            # real_seq_points=self._pop_n_items(self.sampling_seq_list, self.batchSize)
            # norm_cond_points=self._pop_n_items(self.sampling_cond_list, self.batchSize)
            real_cond_points=self._getRealCondition(norm_cond_points)
            
            for idx, real_seq_point in enumerate(real_seq_points):
                real_cond_points[idx]['seq']=real_seq_point
                norm_cond_points[idx]['seq']=real_seq_point
        
        elif self.countSamplingNum < self.samplingNum and (self.countSamplingNum+self.batchSize) > self.samplingNum:
            self.model_logger.info("SeqOpt ({})".format("suggestNextStep-sampling"), "final initial sampling")
            final_experiment_num=self.samplingNum-self.countSamplingNum
            real_seq_points=self._pop_n_items(self.sampling_seq_list, self.countSamplingNum, final_experiment_num)
            norm_cond_points=self._pop_n_items(self.sampling_cond_list, self.countSamplingNum, final_experiment_num)
            # real_seq_points=self._pop_n_items(self.sampling_seq_list, final_experiment_num)
            # norm_cond_points=self._pop_n_items(self.sampling_cond_list, final_experiment_num)
            real_cond_points=self._getRealCondition(norm_cond_points)
            
            for idx, real_seq_point in enumerate(real_seq_points):
                real_cond_points[idx]['seq']=real_seq_point
                norm_cond_points[idx]['seq']=real_seq_point
        
        else: # if self.countSamplingNum satisfy SamplingNum value
            self.model_logger.info("SeqOpt ({})".format("suggestNextStep-AI"), "AI recommended condition")
            norm_seq_points, norm_cond_array=self._suggestAI(self.batchSize)
            # print("norm_cond_array", norm_cond_array)
            # round synthesis condition
            norm_cond_points = [] 
            for i in range(len(norm_cond_array)):
                # norm_cond_params=self._norm_space.array_to_params(norm_cond_array[i])
                # real_cond_params=self._getRealCondition([norm_cond_params])
                # real_cond_array=self._real_space.params_to_array(real_cond_params[0])
                # round_real_cond_array=self._real_space._bin(real_cond_array)
                # round_real_cond_params=self._real_space.array_to_params(round_real_cond_array)
                # round_norm_cond_params=self._getNormalizedCondition([round_real_cond_params])
                # norm_cond_points.append(round_norm_cond_params)
                round_sample= self._norm_space._bin(norm_cond_array[i])
                norm_cond_points.append(self._norm_space.array_to_params(round_sample))
            # print("norm_cond_points", norm_cond_points)
            search_epoch=(len(self._norm_space.res())-self.samplingNum)/self.batchSize

            real_cond_points=self._getRealCondition(norm_cond_points)
            for idx, norm_seq_point in enumerate(norm_seq_points):
                self.model_logger.info("SeqOpt ({})".format("suggestNextStep-AI"), "epoch {}-->next seq:{}, cond:{}".format(search_epoch, norm_seq_point, real_cond_points[idx]))
        
            for idx, norm_seq_point in enumerate(norm_seq_points):
                real_cond_points[idx]['seq']=norm_seq_point.tolist() # seq
                norm_cond_points[idx]['seq']=norm_seq_point.tolist()
        
        return real_cond_points, norm_cond_points

    def registerPoint(self, input_next_points:list, norm_input_next_points:list, property_list:list, input_result_list:list):
        """
        :param input_next_points (dict in list) : [{},{},{}] --> this list has sequence of condition which follow utility function
            ex) ['ucb', 'ucb', 'ucb', 'ucb', 'ei', 'ei', 'es', 'es']
        :param input_result_list (list): [] --> include each result_dict, 
                                            this list has sequence of synthesis condition which called input_next_points
        :return : None
        """
        # search_epoch=(len(self._norm_space.res())-self.samplingNum)/self.batchSize
        norm_input_next_seq_points=[norm_input_next_point["seq"] for norm_input_next_point in norm_input_next_points]
        norm_input_next_params_points=[]
        for norm_input_next_point in norm_input_next_points:
            norm_input_next_point.pop("seq")
            norm_input_next_params_points.append(norm_input_next_point)
        real_input_next_seq_points=[real_input_next_point["seq"] for real_input_next_point in input_next_points]
        real_input_next_params_points=[]
        for real_input_next_point in input_next_points:
            real_input_next_point.pop("seq")
            real_input_next_params_points.append(real_input_next_point)
        
        for process_idx in range(len(input_next_points)):
            optimal_value = input_result_list[process_idx]

            property_dict=property_list[process_idx]
            propertys=list(property_dict.values())

            self._register(space=self._norm_space, seq=norm_input_next_seq_points[process_idx], params=norm_input_next_params_points[process_idx], propertys=propertys, target=optimal_value)
            self._register(space=self._real_space, seq=real_input_next_seq_points[process_idx], params=real_input_next_params_points[process_idx], propertys=propertys, target=optimal_value)

        self.SeqOpt_obj.nn_block.n_observation=len(self._norm_space.res())/len(self.reagent_seqs)*2
        search_epoch=int(len(self._norm_space.res())/self.batchSize)
        best_loss, y_real_pred, best_mae, best_model_state_dict=self._training(search_epoch=search_epoch)

        self.best_loss_list.append(best_loss)
        self.best_mae_list.append(best_mae)
        torch.save(best_model_state_dict, '{}/model_stat_{}.pt'.format(self.TOTAL_MODEL_FOLDER, search_epoch))
        self._drawPlot(search_epoch=search_epoch, y_real_pred=y_real_pred)

    def _training(self,search_epoch):
        self.SeqOpt_obj, best_loss, best_model_state_dict=train_seq_real(self.subject, search_epoch, self.SeqOpt_obj, 
                                                                        self.optimizer, self.model_logger, self._norm_space, self.patience, 
                                                                        self.n_train_epochs, self.device)
        y_real_pred, best_mae=caculateMAE_seq_real(self.subject, search_epoch, self.SeqOpt_obj, self.model_logger, self._norm_space, self.device)

        return best_loss, y_real_pred, best_mae, best_model_state_dict

    def _drawPlot(self, search_epoch, y_real_pred):
        """
        draw y_values scatter plot
        """
        x_values=[]
        for i in range(1, search_epoch+1):
            for _ in range(self.batchSize):
                x_values.append(i)
        y_values=self._norm_space.target
        # print(self.batchSize)
        # print("len(x_values), x_values", len(x_values), x_values)
        # print("len(y_values), y_values", len(y_values), y_values)
        plt.scatter(x_values, y_values, label='y_searched')
        plt.xlabel('Search epochs')
        plt.ylabel('y_searched')
        plt.title("{}_{}".format(self.subject, search_epoch))
        plt.tight_layout()
        plt.savefig('{}/result_searched_y_{}.png'.format(self.TOTAL_RESULT_FOLDER,search_epoch),dpi=300)
        plt.close()
        # CSV 파일로 저장
        df = pd.DataFrame({'Search epochs': x_values, 'y_searched': y_values})
        df.to_csv('{}/result_searched_y_{}.csv'.format(self.TOTAL_RESULT_FOLDER,search_epoch), index=False)
        """
        draw best_y_values csv
        """
        best_y=min(y_values)
        self.best_y_list.append(best_y)
        # CSV 파일로 저장
        df = pd.DataFrame({'Search epochs': [x for x in range(1, len(self.best_y_list)+1)], 'y_searched': self.best_y_list})
        df.to_csv('{}/result_searched_best_y_{}.csv'.format(self.TOTAL_RESULT_FOLDER,search_epoch), index=False)
        """
        draw loss scatter plot
        """
        x_values_loss_mae=[i for i in range(1, search_epoch+1)]
        plt.scatter(x_values_loss_mae, self.best_loss_list, label='Loss')
        plt.xlabel('Search epochs')
        plt.ylabel('Loss')
        # plt.ylim([0, 1]) # 새로 추가 (20250310) --> Evidence SEQ
        plt.title("{}_{}".format(self.subject, search_epoch))
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/result_loss_{}.png'.format(self.TOTAL_RESULT_FOLDER,search_epoch),dpi=300)
        plt.close()
        # CSV 파일로 저장
        df = pd.DataFrame({'Search epochs': x_values_loss_mae, 'Loss': self.best_loss_list})
        df.to_csv('{}/result_loss_{}.csv'.format(self.TOTAL_RESULT_FOLDER,search_epoch), index=False)
        """
        draw mae scatter plot
        """
        plt.scatter(x_values_loss_mae, self.best_mae_list, label='MAE')
        plt.xlabel('Search epochs')
        plt.ylabel('MAE')
        plt.title("{}_{}".format(self.subject, search_epoch))
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/result_mae_{}.png'.format(self.TOTAL_RESULT_FOLDER,search_epoch),dpi=300)
        plt.close()
        # CSV 파일로 저장
        df = pd.DataFrame({'Search epochs': x_values_loss_mae, 'MAE': self.best_mae_list})
        df.to_csv('{}/result_mae_{}.csv'.format(self.TOTAL_RESULT_FOLDER,search_epoch), index=False)

        # 데이터 분리
        y_real = [d[0] for d in y_real_pred]
        y_pred = [d[1] for d in y_real_pred]
        # y=x 그래프 추가
        # scatter plot 그리기
        plt.figure(figsize=(8, 6))
        plt.scatter(y_real, y_pred, color='blue')
        plt.plot(y_real, y_real, color='red', linestyle='--', label='y=x')
        plt.xlabel('y_real')
        plt.ylabel('y_pred')
        plt.title("y_real vs y_pred")
        plt.grid(True)
        plt.savefig('{}/result_y_real_vs_y_pred_{}.png'.format(self.TOTAL_RESULT_FOLDER, search_epoch),dpi=300)
        plt.close()
        # CSV 파일로 저장
        df = pd.DataFrame({'y_real': y_real, 'y_pred': y_pred})
        df.to_csv('{}/result_y_real_vs_y_pred_{}.csv'.format(self.TOTAL_RESULT_FOLDER, search_epoch), index=False)

    def output_space(self, filename):
        """
        [Modified by HJ]

        Outputs complete space as csv file. --> convert normalize condition to real condition
        Simple function for testing
        
        Parameters
        ----------
        dirname (str) :"DB/2022XXXX
        filename : "{}_data" + .csv
        
        Returns
        -------
        None
        """
        total_path="{}/{}.csv".format(self.TOTAL_RESULT_FOLDER, filename)
        if os.path.isdir(self.TOTAL_RESULT_FOLDER) == False:
            os.makedirs(self.TOTAL_RESULT_FOLDER)
        df = pd.DataFrame(data=self._norm_space.params, columns=self._norm_space.keys)
        df['seq'] = self._norm_space.seqs
        df['Target'] = self._norm_space.target
        df.to_csv(total_path, index=False)

    def output_space_realCondition(self, filename):
        """
        [Modified by HJ]

        Outputs complete space as csv file. --> convert normalize condition to real condition
        Simple function for testing
        
        Parameters
        ----------
        dirname (str) :"DB/2022XXXX
        filename : "{}_data" + .csv
        
        Returns
        -------
        None
        """
        total_path="{}/{}_real.csv".format(self.TOTAL_RESULT_FOLDER, filename)
        if os.path.isdir(self.TOTAL_RESULT_FOLDER) == False:
            os.makedirs(self.TOTAL_RESULT_FOLDER)
        df = pd.DataFrame(data=self._real_space.params, columns=self._real_space.keys)
        df['seq'] = self._real_space.seqs
        df['Target'] = self._real_space.target
        df.to_csv(total_path, index=False)

    def output_space_property(self, filename):
        """
        [Modified by HJ]

        Outputs complete space as csv file. --> extract until all property based on real condition
        Simple function for testing
        
        Parameters
        ----------
        dirname (str) :"DB/2022XXXX
        filename : "{}_data" + .csv
        
        Returns
        -------
        None
        """
        try:
            total_path="{}/{}_property.csv".format(self.TOTAL_RESULT_FOLDER, filename)
            if os.path.isdir(self.TOTAL_RESULT_FOLDER) == False:
                os.makedirs(self.TOTAL_RESULT_FOLDER)
            df_property = pd.DataFrame(data=self._real_space.propertys, columns=self._real_space.property_list)
            df = pd.DataFrame(data=self._real_space.params, columns=self._real_space.keys)
            df['seq'] = self._real_space.seqs
            # df=pd.concat([df_2, df_1])
            for property_name in self._real_space.property_list:
                df[property_name]=df_property[property_name]
            df.to_csv(total_path, index=False)
        except:
            print("Wrong path")
            print("self._real_space.property_list:", self._real_space.property_list)
            print("df_property:", df_property)
            print("df:", df)
            pass

    def savedModel(self, filename):
        """
        save ML model to use already fitted model later.
        
        Arguments
        ---------
        directory_path (str)
        filename (str) +.pkl
        
        Returns
        -------
        return None
        """
        fname = os.path.join(self.TOTAL_OBJECT_FOLDER, filename+".pkl")
        if os.path.isdir(self.TOTAL_OBJECT_FOLDER) == False:
            os.makedirs(self.TOTAL_OBJECT_FOLDER)
            time.sleep(3)
            with open(fname, 'wb') as f:
                dill.dump(self, f)
        else:
            with open(fname, 'wb') as f:
                dill.dump(self, f)
                time.sleep(3)
                
    def loadModel(self, filename):
        """
        load ML model to use already fitted model later depending on filename.
        
        Arguments
        ---------
        directory_path (str)
        filename (str)
        
        Returns
        -------
        return loaded_model, model_obj
        """
        fname = os.path.join(self.TOTAL_OBJECT_FOLDER, filename+".pkl")
        with open(fname, 'rb') as f:
            model_obj = dill.load(f)
        return model_obj

if __name__=="__main__":
    algorithm_dict={
        'model': 'SequenceOptimization', 'modelPath': '', 
        'batchSize': 4, 'totalCycleNum': 1, 'verbose': 0, 'randomState': 0, 
        'prange': {
            'BatchSynthesis_AddSolution=AgNO3_Volume': [100, 4000, 50], 
            'BatchSynthesis_AddSolution=NaBH4_Volume': [100, 1000, 50], 
            'BatchSynthesis_AddSolution=PVP55_Volume': [100, 4000, 50], 
            'BatchSynthesis_AddSolution=H2O2_Volume': [100, 4000, 50], 
            'BatchSynthesis_AddSolution=Citrate_Volume': [100, 4000, 50]
            }, 
        'initParameterList': [], 'constraints': [], 
        'sampling': {
            'samplingMethod': 'latin', 
            'samplingNum': 20
            }, 
        'structure': {
            'reagent_list': ['AgNO3', 'NaBH4'], 
            'ps_dim': 4, 
            'output_dim': 1, 
            'nn_n_hidden': 50, 
            'n_train_epochs': 50000, 
            'lr': 0.005, 
            'patience': 1000,
            "device": "cuda:1"
            }, 
        'acq': {
            'acqMethod': 'lcb', 'acqSampler': 'greedy', 
            'acqHyperparameter': {'kappa': 0.05}, 
            'acq_n_samples': 10000
            }, 
        'loss': {
            'lossMethod': 'seqlambdamaxFWHMintensityLoss', 
            'lossTarget': {
                'UV_GetAbs': {
                    'Property': {'lambdamax': 513}, 
                    'Ratio': {'lambdamax': 0.7, 'FWHM': 0.3}
                    }
                }
            }, 
        'subject': 'SeqOpt-AI-1', 
        'group': 'KIST_CSRC', 
        'logLevel': 'DEBUG', 
        'userName': 'HJ', 
        'modeType': 'virtual', 
        'jobID': 0, 
        'jobStatus': 'Waiting...', 
        'temperature': 25, 
        'humidity': '68%', 
        'jobSubmitTime': '2024-08-26 17:29:29'
    }
    seq_opt_obj=NanoChef(algorithm_dict=algorithm_dict)
    # seq_opt_obj.suggestNextStep()
    import pandas as pd
    df_property=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data_property.csv")
    df_real=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data_real.csv")
    df_norm=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data.csv")

    df_property=df_property.drop(columns = ['seq'])
    main_property_list=df_property.to_dict('records')

    main_input_result_list=df_norm["Target"].to_list()
    seq_list=df_norm["seq"]
    seq_list=[eval(seq) for seq in seq_list]

    df_norm=df_norm.drop(columns = ['Target'])  #여러개 열을 삭제할 때.
    df_real=df_real.drop(columns = ['Target'])  #여러개 열을 삭제할 때.

    temp_norm_input_next_points=df_norm.to_dict('records')
    temp_input_next_points=df_real.to_dict('records')
    main_norm_input_next_points=[]
    for norm_input_next_point in temp_norm_input_next_points:
        norm_input_next_point["seq"]=eval(norm_input_next_point['seq'])
        main_norm_input_next_points.append(norm_input_next_point)
    main_input_next_points=[]
    for input_next_point in temp_input_next_points:
        input_next_point["seq"]=eval(input_next_point['seq'])
        main_input_next_points.append(input_next_point)

    # norm_list=[eval(norm) for norm in norm_list]
    # real_list=[eval(real) for real in real_list]

    # input_next_points=[
    #     {"params": param, "seq": seq}
    #     for param, seq in zip(real_list, seq_list)
    # ]
    # norm_input_next_points=[
    #     {"params": param, "seq": seq}
    #     for param, seq in zip(norm_list, seq_list)
    # ]

    # input_next_points=[
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 1000, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 1000, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 1000, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 1000, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 1000
    #         },
    #         "seq":["AgNO3","NaBH4"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 2000, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 2000, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 2000, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 2000, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 2000
    #         },
    #         "seq":["AgNO3","NaBH4"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 3000, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 3000, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 3000, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 3000, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 3000
    #         },
    #         "seq":["NaBH4","AgNO3"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 4000, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 4000, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 4000, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 4000, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 4000
    #         },
    #         "seq":["NaBH4","AgNO3"]
    #     },
    # ]
    # norm_input_next_points=[
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 0.1, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 0.1, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 0.1, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 0.1, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 0.1
    #         },
    #         "seq":["AgNO3","NaBH4"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 0.2, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 0.2, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 0.2, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 0.2, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 0.2
    #         },
    #         "seq":["AgNO3","NaBH4"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 0.3, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 0.3, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 0.3, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 0.3, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 0.3
    #         },
    #         "seq":["NaBH4","AgNO3"]
    #     },
    #     {
    #         "params":{
    #             'BatchSynthesis_AddSolution=AgNO3_Volume': 0.4, 
    #             'BatchSynthesis_AddSolution=NaBH4_Volume': 0.4, 
    #             'BatchSynthesis_AddSolution=H2O_Volume': 0.4, 
    #             'BatchSynthesis_AddSolution=H2O2_Volume': 0.4, 
    #             'BatchSynthesis_AddSolution=Citrate_Volume': 0.4
    #         },
    #         "seq":["NaBH4","AgNO3"]
    #     }
    # ]
    # property_list=[
    #     {'lambdamax': 0.9, 'FWHM': 0.03, 'intensity': 0.07},
    #     {'lambdamax': 0.9, 'FWHM': 0.04, 'intensity': 0.06},
    #     {'lambdamax': 0.9, 'FWHM': 0.05, 'intensity': 0.05},
    #     {'lambdamax': 0.9, 'FWHM': 0.06, 'intensity': 0.04},
    #     ]
    # input_result_list=[-0.1, -0.14, -0.2, -0.3]
    print("input_next_points", main_input_next_points)
    print("norm_input_next_points", main_norm_input_next_points)
    print("property_list", main_property_list)
    print("input_result_list", main_input_result_list)
    seq_opt_obj.registerPoint(main_input_next_points, main_norm_input_next_points, main_property_list, main_input_result_list)
    seq_opt_obj.registerPoint(
        [
            {
                'BatchSynthesis_AddSolution=AgNO3_Volume': 3000, 
                'BatchSynthesis_AddSolution=NaBH4_Volume': 3000, 
                'BatchSynthesis_AddSolution=PVP55_Volume': 3000, 
                'BatchSynthesis_AddSolution=H2O2_Volume': 3000, 
                'BatchSynthesis_AddSolution=Citrate_Volume': 3000,
                "seq":["AgNO3","NaBH4"]
            }
        ], 
        [ 
            {
                'BatchSynthesis_AddSolution=AgNO3_Volume': 0.2, 
                'BatchSynthesis_AddSolution=NaBH4_Volume': 0.2, 
                'BatchSynthesis_AddSolution=PVP55_Volume': 0.2, 
                'BatchSynthesis_AddSolution=H2O2_Volume': 0.2, 
                'BatchSynthesis_AddSolution=Citrate_Volume': 0.2,
                "seq":["AgNO3","NaBH4"]
            }
        ], 
        [{'lambdamax': 512, 'FWHM': 250}], 
        [0.21])
    

    df_property=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data_property.csv")
    df_real=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data_real.csv")
    df_norm=pd.read_csv("Data/SeqOpt-constraints/Result/real/20240919/20240920_103324_9_data.csv")

    df_property=df_property.drop(columns = ['seq'])
    main_property_list=df_property.to_dict('records')

    main_input_result_list=df_norm["Target"].to_list()
    seq_list=df_norm["seq"]
    seq_list=[eval(seq) for seq in seq_list]

    df_norm=df_norm.drop(columns = ['Target'])  #여러개 열을 삭제할 때.
    df_real=df_real.drop(columns = ['Target'])  #여러개 열을 삭제할 때.

    temp_norm_input_next_points=df_norm.to_dict('records')
    temp_input_next_points=df_real.to_dict('records')
    main_norm_input_next_points=[]
    for norm_input_next_point in temp_norm_input_next_points:
        norm_input_next_point["seq"]=eval(norm_input_next_point['seq'])
        main_norm_input_next_points.append(norm_input_next_point)
    main_input_next_points=[]
    for input_next_point in temp_input_next_points:
        input_next_point["seq"]=eval(input_next_point['seq'])
        main_input_next_points.append(input_next_point)
    
    new_seq_opt_obj=NanoChef(algorithm_dict=algorithm_dict)
    print(main_input_next_points)
    main_input_next_points.append({
                'BatchSynthesis_AddSolution=AgNO3_Volume': 3000, 
                'BatchSynthesis_AddSolution=NaBH4_Volume': 3000, 
                'BatchSynthesis_AddSolution=PVP55_Volume': 3000, 
                'BatchSynthesis_AddSolution=H2O2_Volume': 3000, 
                'BatchSynthesis_AddSolution=Citrate_Volume': 3000,
                'seq':["AgNO3","NaBH4"]
            })
    print(main_norm_input_next_points)
    main_norm_input_next_points.append({
                'BatchSynthesis_AddSolution=AgNO3_Volume': 0.2, 
                'BatchSynthesis_AddSolution=NaBH4_Volume': 0.2, 
                'BatchSynthesis_AddSolution=PVP55_Volume': 0.2, 
                'BatchSynthesis_AddSolution=H2O2_Volume': 0.2, 
                'BatchSynthesis_AddSolution=Citrate_Volume': 0.2,
                'seq':["AgNO3","NaBH4"]
            })
    main_property_list.append({'lambdamax': 512, 'FWHM': 250})
    main_input_result_list.append(0.21)
    new_seq_opt_obj.registerPoint(main_input_next_points, main_norm_input_next_points, main_property_list, main_input_result_list)
    # dirname="20240826"
    # filename="20240826_model_1"
    # seq_opt_obj.output_space(dirname+"/loss_norm", filename)
    # seq_opt_obj.output_space_realCondition(dirname+"/loss_real", filename)
    # seq_opt_obj.output_space_property(dirname+"/property", filename)