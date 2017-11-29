#pragma once
#include <QObject>

class MainModel : public QObject {
	Q_OBJECT

public:
	MainModel(QObject * parent = nullptr);
	~MainModel();

private:
	
};
