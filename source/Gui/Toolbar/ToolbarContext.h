#pragma once
#include <QObject>

class ToolbarContext : public QObject {
	Q_OBJECT
public:
	ToolbarContext(QObject * parent = nullptr);
	~ToolbarContext();

private:
	
};
