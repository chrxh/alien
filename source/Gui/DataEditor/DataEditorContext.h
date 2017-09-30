#pragma once

#include <QObject>

#include "Gui/Definitions.h"
#include "Model/Descriptions.h"

class DataEditorContext
	: public QObject
{
	Q_OBJECT
public:
	DataEditorContext(QObject *parent = nullptr);
	virtual ~DataEditorContext() = default;

	Q_SIGNAL void show(bool visible);
};