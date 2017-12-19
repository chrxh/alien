#pragma once

#include <QObject>

#include "Gui/Definitions.h"
#include "Model/Api/Descriptions.h"

class DataEditContext
	: public QObject
{
	Q_OBJECT
public:
	DataEditContext(QObject *parent = nullptr);
	virtual ~DataEditContext() = default;

	Q_SIGNAL void onShow(bool visible);
	Q_SIGNAL void onRefresh();
};