#pragma once

#include "Base/Definitions.h"
#include "ModelBasic/ChangeDescriptions.h"

#include "CudaAccessTOs.cuh"
#include "DefinitionsImpl.h"

class _CudaJob
{
public:
	bool isNotifyFinish() const
	{
		return _notifyFinish;
	}

	string getOriginId() const
	{
		return _originId;
	}

protected:
	_CudaJob(string const& originId, bool notifyFinish) : _originId(originId), _notifyFinish(notifyFinish) { }
	virtual ~_CudaJob() = default;

private:
	string _originId;
	bool _notifyFinish = false;
};

class _GetDataJob 
	: public _CudaJob
{
public:
	_GetDataJob(string const& originId, IntRect const& rect, DataAccessTO const& dataTO)
		: _CudaJob(originId, true), _rect(rect), _dataTO(dataTO) { }

	virtual ~_GetDataJob() = default;

	IntRect getRect() const
	{
		return _rect;
	}

	DataAccessTO getDataTO() const
	{
		return _dataTO;
	}

private:
	DataAccessTO _dataTO;
	IntRect _rect;
};

class _GetDataForImageJob
	: public _GetDataJob
{
public:
	_GetDataForImageJob(string const& originId, IntRect const& rect, DataAccessTO const& dataTO, QImage* targetImage)
		: _GetDataJob(originId, rect, dataTO), _targetImage(targetImage) { }

	virtual ~_GetDataForImageJob() = default;

	QImage* getTargetImage() const
	{
		return _targetImage;
	}

private:
	QImage* _targetImage;
};

class _GetDataForEditJob
	: public _GetDataJob
{
public:
	_GetDataForEditJob(string const& originId, IntRect const& rect, DataAccessTO const& dataTO)
		: _GetDataJob(originId, rect, dataTO) { }

	virtual ~_GetDataForEditJob() = default;

};

class _GetDataForUpdateJob
	: public _GetDataJob
{
public:
	_GetDataForUpdateJob(string const& originId, IntRect const& rect, DataAccessTO const& dataTO, DataChangeDescription const& updateDesc)
		: _GetDataJob(originId, rect, dataTO), _updateDesc(updateDesc) { }

	virtual ~_GetDataForUpdateJob() = default;

	DataChangeDescription const& getUpdateDescription() const
	{
		return _updateDesc;
	}

private:
	DataChangeDescription _updateDesc;
};

class _SetDataJob
	: public _CudaJob
{
public:
	_SetDataJob(string const& originId, bool notifyFinish, IntRect const& rect, DataAccessTO const& dataTO)
		: _CudaJob(originId, notifyFinish), _rect(rect), _dataTO(dataTO) { }

	virtual ~_SetDataJob() = default;

	DataAccessTO getDataTO() const
	{
		return _dataTO;
	}

	IntRect getRect() const
	{
		return _rect;
	}

private:
	DataAccessTO _dataTO;
	IntRect _rect;
};

class _RunSimulationJob
	: public _CudaJob
{
public:
	_RunSimulationJob(string const& originId, bool notifyFinish)
		: _CudaJob(originId, notifyFinish){ }

	virtual ~_RunSimulationJob() = default;
};

class _StopSimulationJob
	: public _CudaJob
{
public:
	_StopSimulationJob(string const& originId, bool notifyFinish)
		: _CudaJob(originId, notifyFinish) { }

	virtual ~_StopSimulationJob() = default;
};

class _CalcSingleTimestepJob
	: public _CudaJob
{
public:
	_CalcSingleTimestepJob(string const& originId, bool notifyFinish)
		: _CudaJob(originId, notifyFinish) { }

	virtual ~_CalcSingleTimestepJob() = default;
};

class _TpsRestrictionJob
	: public _CudaJob
{
public:
	_TpsRestrictionJob(string const& originId, optional<int> tpsRestriction, bool notifyFinish = false)
		: _CudaJob(originId, notifyFinish), _tpsRestriction(tpsRestriction) { }

	virtual ~_TpsRestrictionJob() = default;

	optional<int> getTpsRestriction() const
	{
		return _tpsRestriction;
	}

private:
	optional<int> _tpsRestriction;
};

class _SetSimulationParametersJob
	: public _CudaJob
{
public:
	_SetSimulationParametersJob(string const& originId, SimulationParameters const& parameters, bool notifyFinish = false)
		: _CudaJob(originId, notifyFinish), _parameters(parameters) { }

	virtual ~_SetSimulationParametersJob() = default;

	SimulationParameters const& getSimulationParameters() const
	{
		return _parameters;
	}

private:
	SimulationParameters _parameters;
};
