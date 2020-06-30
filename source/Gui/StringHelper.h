#pragma once

#include "Definitions.h"
#include "Settings.h"

class StringHelper
{
public:
	static QString toString(int value)
	{
		return QString::fromStdString(std::to_string(value));
	}

	static QString toString(double value)
	{
		return QString("%1").arg(value);// QString::fromStdString(std::to_string(value));
	}

	static QString generateFormattedIntString(int value, bool withComma = false)
	{
		QString colorDataStart = "<span style=\"color:" + Const::CellEditDataColor1.name() + "\">";
		QString colorEnd = "</span>";
        if (!withComma) {
            return colorDataStart + QString("%1").arg(value) + colorEnd;
        }
        else {
            QString valueAsString;

            do {
                if (!valueAsString.isEmpty()) {
                    valueAsString = QString(",") + valueAsString;
                }
                if (value >= 1000) {
                    valueAsString = QString("%1").arg(value % 1000, 3) + valueAsString;
                }
                else {
                    valueAsString = QString("%1").arg(value % 1000) + valueAsString;
                }
                value = value / 1000;
            } while (value > 0);
            valueAsString.replace(" ", "0");

            return colorDataStart + valueAsString + colorEnd;
        }
	}

	static QString generateFormattedRealString(qreal r, bool withComma = false)
	{
		QString colorDataStart = "<span style=\"color:" + Const::CellEditDataColor1.name() + "\">";
		QString colorData2Start = "<span style=\"color:" + Const::CellEditDataColor2.name() + "\">";
		QString colorEnd = "</span>";
		bool negativeSign = false;
		if (r < 0.0) {
			r = -r;
			negativeSign = true;
		}
		int i = qFloor(r);
		int re = (r - qFloor(r))*10000.0;
		QString iS = generateFormattedIntString(i, withComma);
		QString reS = QString("%1").arg(re, 4);
		reS.replace(" ", "0");
		if (negativeSign)
			return colorDataStart + "-" + iS + colorEnd + colorData2Start + "." + reS + colorEnd;
		else
			return colorDataStart + iS + colorEnd + colorData2Start + "." + reS + colorEnd;
	}

	static QString ws(int num)
	{
		QString result;
		for (int i = 0; i < num; ++i) {
			result += "&nbsp;";
		}
		return result;
	}
};
