#pragma once

#include "Model/Api/Definitions.h"

class CellComputerCompilerLocal;
class Cell;
class Cluster;

struct CellClusterHash
{
	std::size_t operator()(Cluster* const& s) const;
};
typedef std::unordered_set<Cluster*, CellClusterHash> CellClusterSet;

struct CellHash
{
	std::size_t operator()(Cell* const& s) const;
};
typedef std::unordered_set<Cell*, CellHash> CellSet;
