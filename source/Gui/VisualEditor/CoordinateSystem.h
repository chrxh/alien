#pragma once

#include <QVector2D>

class CoordinateSystem
{
public:
	static QVector2D sceneToModel(QVector2D const &vec);
	static QPointF sceneToModel(QPointF const &p);
	static double sceneToModel(double len);

	static QVector2D modelToScene(QVector2D const &vec);
	static double modelToScene(double len);

};

