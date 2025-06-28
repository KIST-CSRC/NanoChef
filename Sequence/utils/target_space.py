#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [target_space.py] control target space (fitness, lambdmax, FWHM, intensity)
# @Inspiration
    # https://github.com/fmfn/BayesianOptimization
    # https://github.com/CooperComputationalCaucus/kuka_optimizer
# @author   Hyuk Jun Yoo (yoohj9475@kist.re.kr)   
# @version  1_2   
# TEST 2021-11-01
# TEST 2022-04-11

import numpy as np
from skopt.sampler import Grid
from skopt.sampler import Lhs
from skopt.sampler import Sobol
# from skopt.sampler import Random
from skopt.space import Space


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added
    
    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds={},random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.
        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func
        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, new_params):
        self._params=new_params

    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, new_target):
        self._target=new_target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, new_keys):
        self._keys=new_keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        # print("params", params)
        # print("self.keys", self.keys)
        if isinstance(params,list):
            x = []
            for p in params:
                try:
                    assert set(p) == set(self.keys)
                except AssertionError:
                    raise ValueError(
                        "Parameters' keys ({}) do ".format(sorted(params)) +
                        "not match the expected set of keys ({}).".format(self.keys)
                    )
                x.append(np.asarray([p[key] for key in self.keys]))
        else: 
            try:
                assert set(params) == set(self.keys)
            except AssertionError:
                raise ValueError(
                    "Parameters' keys ({}) do ".format(sorted(params)) +
                    "not match the expected set of keys ({}).".format(self.keys)
                )
            x = np.asarray([params[key] for key in self.keys])
        return x

    def array_to_params(self, x):
        if isinstance(x,list):
            params = []
            for param in x:
                try:
                    assert len(param) == len(self.keys)
                except AssertionError:
                    raise ValueError(
                        "Size of array ({}) is different than the ".format(len(x)) +
                        "expected number of parameters ({}).".format(len(self.keys))
                    )
                params.append(dict(zip(self.keys, param)))
        else:
            try:
                    assert len(x) == len(self.keys)
            except AssertionError:
                raise ValueError(
                    "Size of array ({}) is different than the ".format(len(x)) +
                    "expected number of parameters ({}).".format(len(self.keys))
                )
            params = dict(zip(self.keys, x))
        return params

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)
        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params:np.ndarray, target:float):
        """
        Append a point and its target value to the known data.
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        y : float
            target function value
        Raises
        ------
        KeyError:
            if the point is not unique
        Notes
        -----
        runs in ammortized constant time
        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique in continuous space'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params:np.ndarray):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.
        
        Notes
        -----
        If x has been previously seen returns a cached value of y.
        
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        
        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target

    def latin_sample(self ,n_samples:int):
        """
        n_sample : the number of sampling =( the number of samlping cycles * the number of batch_size)
        samplingrng is range of param  ex AgNO3 : [500, 3000], AgNO3 NaBH4[[500,3000],[500,3000]]
        x is samplinglist
        """

        lhs = Lhs(lhs_type="centered", criterion="maximin")
        # print("self._bounds", self._bounds)
        samplingrng = self._bounds.tolist()
        space = Space(samplingrng)
        # space = Space(self._bounds)
        x = lhs.generate(space, n_samples, random_state=self.random_state)
        # print("latin",x)
        
        # if eval says reject, reject = true, break
        return np.array(x)
    
    def latin_sample_constraints(self ,n_samples:int, constraints:dict):
        """
        n_sample : the number of sampling =( the number of samlping cycles * the number of batch_size)
        samplingrng is range of param  ex AgNO3 : [500, 3000], AgNO3 NaBH4[[500,3000],[500,3000]]
        x is samplinglist
        """
        max_total_volume = constraints["totalVolume"]

        lhs = Lhs(lhs_type="centered", criterion="maximin")
        # print("self._bounds", self._bounds)
        samplingrng = self._bounds.tolist()
        space = Space(samplingrng)
        # space = Space(self._bounds)
        # x = lhs.generate(space, n_samples, random_state=self.random_state)
        valid_samples = []
        while len(valid_samples) < n_samples:
            samples = lhs.generate(space, n_samples - len(valid_samples), random_state=self.random_state)
            valid_samples.extend([sample for sample in samples if np.sum(sample) <= max_total_volume])
        # print("latin",x)
        
        # if eval says reject, reject = true, break
        return np.array(x)
    
    def sobol_sample(self ,n_samples:int):
        """
        n_sample : the number of sampling =( the number of samlping cycles * the number of batch_size)
        samplingrng is range of param  ex AgNO3 : [500, 3000], AgNO3 NaBH4[[500,3000],[500,3000]]
        x is samplinglist
        """

        sobol = Sobol()

        samplingrng = self._bounds.tolist()
        space = Space(samplingrng)
        # space = Space(self._bounds)
        x = sobol.generate(space, n_samples, random_state=self.random_state)
        # print(x)
        
        # if eval says reject, reject = true, break
        return np.array(x)

    def grid_sample(self ,n_samples:int):
        """
        Params
        ---------------
        n_sample (int) : the number of sampling =( the number of samlping cycles * the number of batch_size)
            
            - samplingrng (list) = range of param  ex AgNO3 : [500, 3000], AgNO3 NaBH4[[500,3000],[500,3000]]
            - x (list) = samplinglist

        Return
        ---------------
        np.array(x)
        """
        grid = Grid(border="include", use_full_layout=True, append_border="exclude")

        samplingrng = self._bounds.tolist()
        # print(samplingrng)

        space = Space(samplingrng)
        x = grid.generate(space, n_samples)
        return np.array(x)

    def random_sample(self ,constraints=[]):
        # TODO: support integer, category, and basic scipy.optimize constraints
        # print("self.dim : ", self.dim) ==> 차원 수
        data = np.empty((1,self.dim))
        reject = True
        while reject:
            for col, (lower, upper) in enumerate(self._bounds):
                data.T[col] = self.random_state.uniform(lower, upper, size=1)
            #    data.T[col] = lower+self.random_state.random(size=1)*(upper-lower)
            reject = False
            for constraint in constraints:
                # if eval says reject, reject = true, break
                if constraint['fun'](data.ravel())<0:
                    reject = True
                break
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds
        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]


class DiscreteSpace(TargetSpace):
    '''
    Holds the param-space coordinates (X) and target values (Y) in the discretized space. 
    This mirrors TargetSpace but supers methods to consider the floor value of discretized bins.
    That is, a prange (-5,5,.5) will register 1.3 as 1.0 in the cache but as 1.3 in the parameters list. 
    Allows for constant-time appends while ensuring no duplicates are added
    '''
    
    def __init__(self, target_func, prange, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            maximum, and step values.
        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator

        """
        
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(prange)
        
        # Get associated pbounds for TargetSpace()
        self._pbounds = {item[0] :(item[1][:2]) for item in sorted(prange.items(), key=lambda x: x[0])} 
            # item --> [100,3000,50]
            # item[1][:2] --> [100,3000,50]
        
        # Create an array with parameters steps
        self._steps = np.array(
            [item[1][-1] for item in sorted(prange.items(), key=lambda x: x[0])],
            dtype=np.float
            )
        # --> self._steps= [50. 50.]
        
        # keep track of unique points we have seen so far
        self._discrete_cache = {}
        
        super(DiscreteSpace, self).__init__(target_func=target_func,
                                            pbounds=self._pbounds,
                                            random_state=random_state,
                                            )
        
    @property
    def steps(self):
        return self._steps
    
    def _bin(self,x):
        """
        round number based on self._steps for each parameter
        >>> x = [220, 2550]
        >>> binned_output = example._bin(x)
        >>> print(binned_output)
            [ 200. 2550.]
        """
        # TODO: clean using modulo 
        binned  = np.empty((self.dim,1))
        for col, (lower, upper) in enumerate(self._bounds):           
            # _bounds = [ [100,3000] , [100,3000]]
            # col = 0, 1 ....
            # lower = 100
            # upper = 3000
            binned[col] = np.floor((x[col]-lower)/self._steps[col])*self._steps[col]+lower # np.floor : 버림 함수
        # print("!!",binned.ravel())
        return binned.ravel()
    
    def __contains__(self,x):
        return(_hashable(self._bin(x))) in self._discrete_cache
    
    def probe_discrete(self, params):    
        """
        Checks discrete cache for x and returns a cached value of y.
        
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        
        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._discrete_cache[_hashable(x)]
        except KeyError:
            raise
        return target

    def register(self, params, target, verbose=False):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        y : float
            target function value
        """

        x = self._as_array(params)
        if x in self and verbose:
            print('Data point {} is not unique. \n(Discrete value {})'.format(x,self._bin(x)))
        # Insert data into unique dictionary
        self._discrete_cache[_hashable(self._bin(x))] = target
        self._cache[_hashable(x.ravel())] = target
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])


class SeqDiscreteSpace(DiscreteSpace):
    '''
    Holds the param-space coordinates (X) and target values (Y) in the discretized space. 
    This mirrors TargetSpace but supers methods to consider the floor value of discretized bins.
    That is, a prange (-5,5,.5) will register 1.3 as 1.0 in the cache but as 1.3 in the parameters list. 
    Allows for constant-time appends while ensuring no duplicates are added
    '''
    
    def __init__(self, prange, target_condition_dict, target_func=None, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            maximum, and step values.
        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator

        """
        self._seqs=[]
        self.property_list=[]
        for action_dict in target_condition_dict.values():
            aciton_property_list=list(action_dict["Ratio"].keys())
            self.property_list.extend(aciton_property_list)
        self._propertys = np.empty(shape=(0, len(self.property_list)))
        
        super(SeqDiscreteSpace, self).__init__(
            target_func=None,
            prange=prange,
            random_state=random_state,
        )
        
    @property
    def seqs(self):
        return self._seqs
    
    @seqs.setter
    def seqs(self, new_seqs):
        self._seqs=new_seqs

    @property
    def propertys(self):
        return self._propertys
    
    @propertys.setter
    def propertys(self, new_propertys):
        self._propertys=new_propertys

    @property
    def propertys_dim(self):
        return len(self.property_list)
    
    def propertys_to_array(self, propertys):
        if isinstance(propertys,list):
            x = []
            for p in propertys:
                try:
                    assert set(p) == set(self.property_list)
                except AssertionError:
                    raise ValueError(
                        "Parameters' keys ({}) do ".format(sorted(propertys)) +
                        "not match the expected set of keys ({}).".format(self.property_list)
                    )
                x.append(np.asarray([p[key] for key in self.property_list]))
        else: 
            try:
                assert set(propertys) == set(self.property_list)
            except AssertionError:
                raise ValueError(
                    "Parameters' keys ({}) do ".format(sorted(propertys)) +
                    "not match the expected set of keys ({}).".format(self.property_list)
                )
            x = np.asarray([propertys[key] for key in self.property_list])
        return x

    def array_to_propertys(self, propertys):
        if isinstance(x,list):
            propertys = []
            for property_value in propertys:
                try:
                    assert len(property_value) == len(self.property_list)
                except AssertionError:
                    raise ValueError(
                        "Size of array ({}) is different than the ".format(len(x)) +
                        "expected number of propertys ({}).".format(len(self.property_list))
                    )
                propertys.append(dict(zip(self.property_list, property_value)))
        else:
            try:
                assert len(x) == len(self.property_list)
            except AssertionError:
                raise ValueError(
                    "Size of array ({}) is different than the ".format(len(x)) +
                    "expected number of propertys ({}).".format(len(self.property_list))
                )
            propertys = dict(zip(self.property_list, propertys))
        return propertys
    
    def _as_array_propertys(self, propertys):
        try:
            propertys = np.asarray(propertys, dtype=float)
        except TypeError:
            propertys = self.propertys_to_array()
        propertys = propertys.ravel()
        try:
            assert propertys.size == self.propertys_dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(propertys)) +
                "expected number of propertys ({}).".format(len(self.property_list))
            )
        return propertys

    def register(self, seq:list, params:np.ndarray, propertys:np.ndarray, target:float, verbose=False):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        seq : list
            ["AgNO3", "NaBH4]
        params : ndarray
            a single point, with len(x) == self.dim
        target : float
            target function value
        """

        x = self._as_array(params)
        y = self._as_array_propertys(propertys)
        if x in self and verbose:
            print('Data point {} is not unique. \n(Discrete value {})'.format(x,self._bin(x)))
        # Insert data into unique dictionary
        self._discrete_cache[_hashable(self._bin(x))] = target
        self._cache[_hashable(x.ravel())] = target

        self._seqs.append(seq)
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._propertys = np.concatenate([self._propertys, y.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])
    
    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]
        propertys = [dict(zip(self.property_list, p)) for p in self.propertys]

        return [
            {"target": target, "params": param, "seq": seq, "property": property_value}
            for target, param, seq, property_value in zip(self.target, params, self.seqs, propertys)
        ]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np 
    lhs_type_list=["classic", "centered"]
    criterion_list=["ratio","correlation","maximin"]
    for lhs_type in lhs_type_list:
        for criterion in criterion_list:
            # lhs = Lhs(lhs_type="centered", criterion="correlation")
            # lhs = Lhs(lhs_type="centered", criterion="maximin")
            lhs = Lhs(lhs_type=lhs_type, criterion=criterion)

            space = Space([[100,3000],[100,3000]])
            # space = Space(self._bounds)
            # data = lhs.generate(space, 20, random_state=3)

            # max_total_volume=3000
            # data = []
            # while len(data) < 20:
            #     samples = lhs.generate(space, 20 - len(data), random_state=0)
            #     data.extend([sample for sample in samples if np.sum(sample) <= max_total_volume])
            #     print(data)
            #     print(len(data))
            data = lhs.generate(space, 20, random_state=0)
            data=np.array(data)

            # x와 y로 데이터 분리
            x = [point[0] for point in data]
            y = [point[1] for point in data]

            # 데이터 포인트의 수
            n_points = len(data)

            # 무지개 색상 컬러맵 설정
            colors = plt.cm.rainbow(np.linspace(0, 1, n_points))

            # 스캐터 플롯
            plt.scatter(x, y, c=colors, s=100, edgecolor='black')

            # 그래프 타이틀과 레이블 설정
            plt.title("{}_{}".format(lhs_type, criterion))
            plt.xlabel("X")
            plt.ylabel("Y")

            # 컬러바 추가 (선택사항)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n_points))
            plt.colorbar(sm, ticks=range(n_points), label='Data Point Index')

            # 그래프 보여주기
            plt.savefig("test_{}_{}_constraints.png".format(lhs_type, criterion), dpi=300)

            plt.close()

