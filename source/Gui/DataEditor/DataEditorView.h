#pragma once
#include <QObject>

class DataEditorView
	: public QObject
{
	Q_OBJECT
public:
	DataEditorView(QObject * parent = nullptr);
	~DataEditorView();

private:
	
};
