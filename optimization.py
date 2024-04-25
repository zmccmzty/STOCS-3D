
import numpy as np
#import pydrake.all as pd
#from pydrake.autodiffutils import AutoDiffXd
from typing import List,Tuple,Dict,Optional,Union,Callable,Any
FloatOrVector = Union[float,np.array]

class Variable:
    """Represents an optimization variable. It can be a scalar, vector,
    or matrix.

    Either value or shape must be provided. If value is provided, shape
    will be inferred from it. If both are provided, they must match shape.

    You may use the [] operator to index into a variable when assigning
    arguments to functions. This will return a new Variable object that
    represents the indexed variable.
    
    Args:
        name (str): The name of the variable.
        value (float or np.array, optional): The current value of the variable.
        shape (Tuple[int], optional): The shape of the variable.
        lb (float or np.array, optional): The lower bound of the variable.
        ub (float or np.array, optional): The upper bound of the variable.
        description (str, optional): A description of the variable.
    """
    def __init__(self, name : str,
                 value : FloatOrVector=None,
                 shape : Tuple[int]=None,
                 lb : FloatOrVector=None,
                 ub : FloatOrVector=None,
                 description : str=None):
        self.name = name
        self.value = value
        self.shape = shape
        if self.value is not None:
            shape = np.shape(self.value)
            if self.shape is not None:
                assert shape == self.shape,"Both value and shape provided, but they don't match"
            self.shape = shape
        self.lb = lb
        self.ub = ub
        self.description = description
        self.solver_encoding = None
        self.solver_impl = None
    
    def set(self,value : FloatOrVector):
        if self.shape is not None:
            assert value.shape == self.shape
        self.value = value
    
    def get(self) -> FloatOrVector:
        assert all(self.value is not None),f"Variable {self.name} value not set"
        return self.value

    def lower_bound(self):
        """Returns the lower bound of the variable. If not set, returns
        negative infinity for each element of the variable."""
        if self.lb is not None:
            return self.lb
        return np.full(self.shape,-np.inf)

    def upper_bound(self):
        """Returns the upper bound of the variable. If not set, returns
        positive infinity for each element of the variable."""
        if self.lb is not None:
            return self.lb
        return np.full(self.shape,np.inf)

    def __getitem__(self,index):
        return _IndexedVariable(self,index)


class _IndexedVariable:
    def __init__(self,v:Variable,index):
        self.parent = v
        self.index = index
        self.name = f'{v.name}[{index}]'
        self.description = f'{v.description}[{index}]'

    def set(self,value : FloatOrVector):
        self.parent.get()[self.index] = value
    
    def get(self) -> FloatOrVector:
        return self.parent.get()[self.index]
    
    def lower_bound(self):
        return self.parent.lower_bound()[self.index]

    def upper_bound(self):
        return self.parent.upper_bound()[self.index]


class Function:
    """Represents a function that takes a set of variables as input.

    The function can take any number of arguments, and within optimization
    it is run as if you called `func(*[v.get() for v in variables])`.  You
    can also specify pre_args and post_args, which are passed to func before
    and after the variables, respectively.

    For Drake autodiff to work, the function must be written to accept
    AutoDiffXd objects as input, i.e., uses plain arithmetic and numpy
    functions.  You may not call C++ libraries (e.g., Klampt geometry
    functions).  Klampt vectorops, so3, and se3 functions are supported.
    
    Args:
        func (Callable): The function to evaluate.
        variables (Union[Variable,Tuple[Variable]]): The variables to pass to the function.
        pre_args (Tuple[FloatOrVector], optional): Arguments to pass before the variables.
        post_args (Tuple[FloatOrVector], optional): Arguments to pass after the variables.
        name (str, optional): The name of the function.
        description (str, optional): A description of the function.
    """
    def __init__(self, func : Callable,
                 variables : Union[Variable,Tuple[Variable]],
                 pre_args : Tuple[FloatOrVector]=None,
                 post_args : Tuple[FloatOrVector]=None,
                 name : str=None,
                 description : str=None):
        self.func = func
        self.variables = [variables] if isinstance(variables,Variable) else variables
        self.pre_args = list(pre_args) if pre_args is not None else []
        self.post_args = list(post_args) if post_args is not None else []
        self.name = name
        self.description = description
        self.solver_encoding = None
        self.solver_impl = None

    def shape(self):
        """Returns the shape of the function output.  By default, this will
        return the shape of the output of func when called with the current
        variables.  If func returns a scalar, this will return ()."""
        return np.shape(self.__call__())
        
    def __call__(self,*args):
        if args is None:
            args = [v.get() for v in self.variables]
        return self.func(*(self.pre_args+args+self.post_args))


class ObjectiveFunction(Function):
    """Simple wrapper that represents an objective function.  The function
    should return a scalar value."""
    def __init__(self, func : Callable,
                 variables : Union[Variable,Tuple[Variable]],
                 pre_args : Tuple[FloatOrVector] = None,
                 post_args : Tuple[FloatOrVector] = None,
                 name : str='objective',
                 description : str=None):
        super().__init__(func,variables,pre_args,post_args,name=name,description=description)
        
    def shape(self):
        return ()


class ConstraintFunction(Function):
    """A function that will be constrained during optimization.  Can either
    be an inequality or an equality constraint.  If lb and ub are set, it is
    an inequality constraint.  If rhs is set, it is an equality constraint.
    """
    def __init__(self, func : Callable,
                 variables : Union[Variable,Tuple[Variable]],
                 lb : FloatOrVector=None,
                 ub : FloatOrVector=None,
                 rhs : FloatOrVector=None,
                 pre_args : Tuple[FloatOrVector] = None,
                 post_args : Tuple[FloatOrVector] = None,
                 name : str='constraint',
                 description : str=None):
        super().__init__(func,variables,pre_args,post_args,name=name,description=description)
        self.lb = lb
        self.ub = ub
        if rhs is not None:
            assert lb == None and ub == None,"Cannot set both rhs and lb/ub"
            self.lb = rhs
            self.ub = rhs
            self._equality = True
        else:
            self._equality = np.all(self.lower_bound() == self.upper_bound())
        assert self.lb is not None or self.ub is not None,"Must set at least one of lb or ub"

    def lower_bound(self):
        if self.lb is not None:
            return self.lb
        return np.full(-np.inf,self.shape())
    
    def upper_bound(self):
        if self.ub is not None:
            return self.ub
        return np.full(np.inf,self.shape())

    def equality(self) -> bool:
        return self._equality
    
    def residual(self):
        """Returns the residual of the constraint.  If the constraint is an
        equality constraint, returns the difference between the function
        output and the rhs.  If the constraint is an inequality constraint,
        returns the violation of the function output vs the bounds."""
        res = self.__call__()
        if self.equality():
            return res - self.lb
        err = None
        if self.ub is not None:
            err = np.maximum(res-self.ub,0)
        if self.lb is not None:
            err = np.maximum(-res+self.lb,0)
        return err


class NonlinearProgramSolver(object):
    """Binds a set of variables, constraints, and an objective function to a
    Drake MathematicalProgram, and solves it using a nonlinear solver.

    More convenient than Drake's API if you have functions that take multiple
    vector variables, since it handles the encoding and decoding of the
    variables for you.

    Attributes:
        variables: A dictionary of Variable objects.
        objective: An ObjectiveFunction object.
        constraints: A dictionary of ConstraintFunction objects.
        prog: The Drake MathematicalProgram object.
    """
    def __init__(self):
        self.variables = {}   # type: Dict[Any,Variable]
        self.objective = None # type: ObjectiveFunction
        self.constraints = {} # type: Dict[Any,ConstraintFunction]
        self.prog = None      # type: pd.MathematicalProgram

    def self_check(self):
        """Checks that all variables, constraints, and the objective are
        properly set up."""
        assert self.objective is not None,"Objective not set"
        for k,v in self.variables.items():
            assert v.value is not None,f"Variable {k} value not set {v.description if v.description is not None else ''}"
        assert self.objective is not None,"Objective not set"
        try:
            self.objective()
        except Exception:
            print("Exception raised while trying to evaluate objective",self.objective.description)
            raise
        for k,c in self.constraints.items():
            assert c.lb is not None or c.ub is not None,f"Constraint {k} bounds not set"
            try:
                c()
            except Exception:
                desc = c.description if c.description is not None else ''
                print("Exception raised while trying to evaluate constraint",k,desc)
                raise
            for v in c.variables:
                assert v in self.variables,f"Constraint {k} has variable {v} not in variables dict"

    def setup_drake_program(self):
        prog = pd.MathematicalProgram()
        self.prog = prog
        for k,v in self.variables.items():
            if len(v.shape)==0:
                v.solver_impl = prog.NewContinuousVariable(name=v.name)
            elif len(v.shape)==1:
                v.solver_impl = prog.NewContinuousVariables(v.shape[0],name=v.name)
            elif len(v.shape)==2:
                v.solver_impl = prog.NewContinuousVariables(rows=v.shape[0],cols=v.shape[1],name=v.name)
            if v.lb is not None or v.ub is not None:
                prog.AddBoundingBoxConstraint(v.lower_bound(),v.upper_bound(),v.solver_impl) 
            if v.value is not None:
                prog.SetInitialGuess(v.solver_impl, v.value)

        for k,func in self.constraints.items():
            flat_func,flat_vars = self._wrap(func)
            c = self.prog.AddConstraint(flat_func,func.lower_bound(),func.upper_bound(),flat_vars)
            if func.description:
                c.evaluator().set_description(func.description)

        # Objective function
        flat_func,flat_vars = self.wrap(self.objective)
        c = self.prog.AddCost(flat_func,flat_vars)
        if self.objective.description:
            c.evaluator().set_description(self.objective.description)

    def solve(self, max_iters=100,
              major_feasibility_tol=1e-4,
              major_optimality_tol=1e-4,
              tmp_file:str=None) -> Dict[Any,FloatOrVector]:
        """Solves the optimization problem starting from the current value of
        the variables.  The return value is a variable dictionary and the 
        current variable values are NOT modified.
        """
        if self.prog is None:
            self.setup_drake_program()
        prog = self.prog
        solver = pd.SnoptSolver()
        snopt = solver.solver_id()
        prog.SetSolverOption(snopt, "Major Iterations Limit", max_iters)
        prog.SetSolverOption(snopt, "Major Feasibility Tolerance", major_feasibility_tol)
        prog.SetSolverOption(snopt, "Major Optimality Tolerance", major_optimality_tol)

        if tmp_file is not None:
            prog.SetSolverOption(snopt, 'Print file', tmp_file)
        result = solver.Solve(prog)

        #parse solution
        res_dict = {}
        for k,v in self.variables.items():
            res_dict[k] = v.value = result.GetSolution(v.solver_impl)
        return res_dict

    def evaluate(self) -> Dict[Any,FloatOrVector]:
        """Evaluates all the variables, objectives, and constraints at the
        current state. 
        """
        res = {}
        vars = {}
        varbounds = {}
        constraints = {}
        constraintbounds = {}
        res['objective'] = self.objective()
        for k,v in self.variables.items():
            if v.value is not None:
                vars[k] = v.value
            if v.lb is not None or v.ub is not None:
                varbounds[k] = v.lower_bound(),v.upper_bound()
        for k,c in self.constraints.items():
            constraints[k] = c()
            constraintbounds[k] = c.lower_bound(),c.upper_bound()
        res['variables'] = vars
        res['variable_bounds'] = varbounds
        res['constraints'] = constraints
        res['constraint_bounds'] = constraintbounds
        return res

    def get_var_dict(self) -> Dict[Any,FloatOrVector]:
        """Returns a dictionary of variable names to their current values."""
        return {k:v.value for k,v in self.variables.items()}

    def set_var_dict(self,values:Dict[Any,FloatOrVector]) -> None:
        """Sets the values of the variables from a dictionary."""
        for k,v in values.items():
            self.variables[k].set(v)

    def _wrap(self,func:Function) -> Tuple[Callable,np.ndarray]:
        """Automatically encodes and decodes heterogeneous variables to a
        function into a Drake function and list of variables."""
        all_scalar = True
        concat_list = []
        size_list = []
        vars = [v.solver_impl for v in func.variables]
        for v in vars:
            if isinstance(v,pd.Variable):
                concat_list.append([v])
                size_list.append(1)
            else:
                if len(v.shape) == 1:
                    concat_list.append(v)
                    size_list.append(len(v))
                elif len(v.shape) == 2:
                    concat_list.append(v.flatten())
                    size_list.append(v.shape[0]*v.shape[1])
                else:
                    raise NotImplementedError("Only scalar, 1D, and 2D variables are supported")
                all_scalar = False
        if all_scalar:
            # just pass through
            def corrected_func(x):
                res = func(x)
                if not hasattr(res,'__iter__'):
                    return [res]
                return res
            return corrected_func,vars
        else:
            split_list = np.cumsum(size_list)[:-1]
            def flattened_func(x):
                x_split = np.split(x,split_list)
                for i in range(len(x_split)):
                    if isinstance(vars[i].solver_impl,pd.Variable):
                        x_split[i] = x_split[i][0]
                    elif len(vars[i].shape)==2:
                        x_split[i] = x_split[i].reshape(vars[i].shape)
                res = func(*x_split)
                if not hasattr(res,'__iter__'):
                    return [res]
                return res
            return flattened_func,np.concatenate(concat_list)
