#pragma once

#include <QObject>

#include "Gui/Definitions.h"
#include "ModelInterface/Descriptions.h"

class DataEditContext
	: public QObject
{
	Q_OBJECT
public:
	DataEditContext(QObject *parent = nullptr);
	virtual ~DataEditContext() = default;

	Q_SIGNAL void show(bool visible);
	Q_SIGNAL void refresh();
};