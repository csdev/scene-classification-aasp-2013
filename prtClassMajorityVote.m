classdef prtClassMajorityVote < prtAction
    %prtClassMajorityVote Insert description of class here
    %
    
    properties (SetAccess=private)
        name = 'MajorityVote'                  % Insert the name of the classifier
        nameAbbreviation = 'MajorityVote'      % A short abbreviation of the name
		isNativeMary = true;          	      % Change to true to create an M-ary classifier
    end
    
    properties (SetAccess=protected)
        isSupervised = true;
        isCrossValidateValid = true;
    end
    
    properties  
       % Place definitions of properties that your classifier requires here
       baseClassifier = [];
       trainingLimitPerClass = 0;
       rejectLowLikelihoodInstances = false;
    end
    
    methods
        
        %Define a constructor
        function Obj = prtClassMajorityVote(varargin)
            %Uncomment this line to enable parameter/value setting of public
            %properties for your classifier
            % Obj = prtUtilAssignStringValuePairs(Obj,varargin{:});
        end
        
        % Put additional methods here (set, get methods, and any other
        % methods your classifier requires)
    end
    
    methods (Access=protected, Hidden = true)
        
        function Obj = trainAction(Obj,DataSet)
            % Place code here to perform classifier training.  This
            % function should set the properties of Obj so that in
            % runAction the classifier can output decision statistics
            if (isempty(Obj.baseClassifier))
                error('base classifier not set');
            end
            
            ds = prtDataSetClass;
            ds = ds.setX(DataSet.getExpandedData());
            ds = ds.setY(DataSet.getExpandedTargets());
            
            if (Obj.trainingLimitPerClass)
                ds = ds.bootstrapByClass(Obj.trainingLimitPerClass);
            end
            
            Obj.baseClassifier = Obj.baseClassifier.train(ds);
        end
        
        function DataSetOut = runAction(Obj,DataSetIn)
            % Place code here to perform classifier running.  This
            % function should set output either:
            %
            %   1) A DataSet with observations of size
            %   DataSetIn.nObservations x 1 (for classifiers with
            %   isNativeMary set to "false").  Higher values should
            %   correspond to data points in DataSetIn which are more
            %   likely to correspond to the positive hypothesis.  Note that 
            %   for binary classification problems, the output of class.run 
            %   should always contain exactly one column.
            %
            %   Or:
            %
            %   2) A DataSet with observations of size
            %   DataSetIn.nObservations x nClasses, where each column of
            %   the output DataSet represents the likelihood that each
            %   observation belongs to the corresponding element of
            %   nClasses
            %
            %   If yOut is a vector or matrix containing the output values,
            %   the DataSetOut can be created with the following line of
            %   code:
            %   DataSetOut = DataSetIn.setObservations(yOut);
            
            result = zeros(DataSetIn.nBags, 1);
            
            for bagIndex = 1:DataSetIn.nBags;
                ds = prtDataSetClass;
                ds = ds.setX(DataSetIn.data(bagIndex).data);
                dsOut = Obj.baseClassifier.run(ds);
                
                if (Obj.rejectLowLikelihoodInstances)
                    [likelihood, decision] = max(dsOut.getX(), [], 2);
                    likelihoodAvg = mean(dsOut.getX(), 2);
                    likelihoodStd = std(dsOut.getX(), 0, 2);
                    
                    confDecision = decision;
                    confDecision(likelihood < likelihoodAvg + 1.28155*likelihoodStd) = NaN;
                    m = mode(confDecision);
                    
                    if (isnan(m))
                        result(bagIndex) = mode(decision);
                        warning('no confident decision can be made for bag %d', bagIndex);
                    else
                        result(bagIndex) = m;
                    end
                else
                    result(bagIndex) = mode(dsOut.getX());
                end
            end
            
            DataSetOut = prtDataSetClass;
            DataSetOut = DataSetOut.setX(result);
        end
        
    end % methods
    
end % classdef
