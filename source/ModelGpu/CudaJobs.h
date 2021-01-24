#pragma once

#include <QImage>

#include "Base/Definitions.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/ExecutionParameters.h"

#include "AccessTOs.cuh"
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

class _ClearDataJob
    : public _CudaJob
{
public:
    _ClearDataJob(string const& originId)
        : _CudaJob(originId, false) { }

    virtual ~_ClearDataJob() = default;
};

class _GetMonitorDataJob
    : public _CudaJob
{
public:
    _GetMonitorDataJob(string const& originId)
        : _CudaJob(originId, true) { }

    virtual ~_GetMonitorDataJob() = default;

    void setMonitorData(MonitorData const& monitorData)
    {
        _monitorData = monitorData;
    }

    MonitorData getMonitorData()
    {
        return _monitorData;
    }

private:
    MonitorData _monitorData;
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

class _GetPixelImageJob
	: public _CudaJob
{
public:
    _GetPixelImageJob(string const& originId, IntRect const& rect, QImagePtr const& targetImage, std::mutex& mutex)
		: _CudaJob(originId, true), _targetImage(targetImage), _mutex(mutex)
    {
        auto imageSize = targetImage->size();

        IntVector2D upperLeft = { std::max(0, rect.p1.x), std::max(0, rect.p1.y) };
        _rect = { upperLeft, {upperLeft.x + imageSize.width() - 1, upperLeft.y + imageSize.height() - 1 } };
    }

    virtual ~_GetPixelImageJob() = default;

    IntRect getRect() const
    {
        return _rect;
    }

    QImagePtr getTargetImage() const
	{
		return _targetImage;
	}

    std::mutex& getMutex()
    {
        return _mutex;
    }

private:
    IntRect _rect;
    QImagePtr _targetImage;
    std::mutex& _mutex;
};

class _GetVectorImageJob
    : public _CudaJob
{
public:
    _GetVectorImageJob(string const& originId, IntRect const& rect, int zoom, QImagePtr const& targetImage, std::mutex& mutex)
        : _CudaJob(originId, true), _zoom(zoom), _targetImage(targetImage), _mutex(mutex), _rect(rect)
    {
    }

    virtual ~_GetVectorImageJob() = default;

    IntRect getRect() const
    {
        return _rect;
    }

    int getZoom() const
    {
        return _zoom;
    }

    QImagePtr getTargetImage() const
    {
        return _targetImage;
    }

    std::mutex& getMutex()
    {
        return _mutex;
    }

private:
    IntRect _rect;
    int _zoom;
    QImagePtr _targetImage;
    std::mutex& _mutex;
};

class _UpdateDataJob
	: public _CudaJob
{
public:
	_UpdateDataJob(
        string const& originId, IntRect const& rect, DataAccessTO const& dataTO, 
        DataChangeDescription const& updateDesc, SimulationParameters const& parameters)
    : _CudaJob(originId, true), _rect(rect), _dataTO(dataTO), _updateDesc(updateDesc), _parameters(parameters) { }

	virtual ~_UpdateDataJob() = default;

    IntRect getRect() const
    {
        return _rect;
    }

    DataAccessTO getDataTO() const
    {
        return _dataTO;
    }

	DataChangeDescription const& getUpdateDescription() const
	{
		return _updateDesc;
	}

    SimulationParameters const& getSimulationParameters() const
    {
        return _parameters;
    }

private:
	DataChangeDescription _updateDesc;
    SimulationParameters _parameters;

    DataAccessTO _dataTO;
    IntRect _rect;
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

class _SetExecutionParametersJob
    : public _CudaJob
{
public:
    _SetExecutionParametersJob(
        string const& originId,
        ExecutionParameters const& parameters,
        bool notifyFinish = false)
        : _CudaJob(originId, notifyFinish)
        , _parameters(parameters)
    {}

    virtual ~_SetExecutionParametersJob() = default;

    ExecutionParameters const& getSimulationExecutionParameters() const
    {
        return _parameters;
    }

private:
    ExecutionParameters _parameters;
};

class _SelectDataJob
    : public _CudaJob
{
public:
    _SelectDataJob(string const& originId, IntVector2D const& pos)
        : _CudaJob(originId, true), _pos(pos) { }

    virtual ~_SelectDataJob() = default;

    IntVector2D getPosition() const
    {
        return _pos;
    }

private:
    IntVector2D _pos;
};

class _DeselectDataJob
    : public _CudaJob
{
public:
    _DeselectDataJob(string const& originId)
        : _CudaJob(originId, true){ }

    virtual ~_DeselectDataJob() = default;
};

class _PhysicalActionJob
    : public _CudaJob
{
public:
    _PhysicalActionJob(string const& originId, PhysicalAction const& action)
        : _CudaJob(originId, false), _action(action) { }

    virtual ~_PhysicalActionJob() = default;

    PhysicalAction getAction()
    {
        return _action;
    }

private:
    PhysicalAction _action;
};
