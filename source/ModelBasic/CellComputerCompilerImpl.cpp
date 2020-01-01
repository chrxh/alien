#include "SymbolTable.h"
#include "SimulationParameters.h"
#include "CompilerHelper.h"
#include "CellComputerCompilerImpl.h"

namespace
{
	enum class CompilerState {
		LOOKING_FOR_INSTR_START,
		LOOKING_FOR_INSTR_END,
		LOOKING_FOR_OP1_START,
		LOOKING_FOR_OP1_END,
		LOOKING_FOR_SEPARATOR,
		LOOKING_FOR_COMPARATOR,
		LOOKING_FOR_OP2_START,
		LOOKING_FOR_OP2_END
	};

	struct InstructionUncoded {
		bool readingFinished = false;
		QString name;
		QString operand1;
		QString operand2;
		QString comp;
	};

	bool isNameChar(const QChar& c)
	{
		return c.isLetterOrNumber() || (c == ':');
	}

	bool gotoNextStateAndReturnSuccess(CompilerState &state, QChar &currentSymbol, InstructionUncoded& instruction, int bytePos, int codeSize)
	{
		switch (state) {
		case CompilerState::LOOKING_FOR_INSTR_START: {
			if (currentSymbol.isLetter()) {
				state = CompilerState::LOOKING_FOR_INSTR_END;
				instruction.name = currentSymbol;
			}
		}
		break;
		case CompilerState::LOOKING_FOR_INSTR_END: {
			if (!currentSymbol.isLetter()) {
				if ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif"))
					instruction.readingFinished = true;
				else
					state = CompilerState::LOOKING_FOR_OP1_START;
			}
			else {
				instruction.name += currentSymbol;
				if ((bytePos + 1) == codeSize && ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif")))
					instruction.readingFinished = true;
			}
		}
		break;
		case CompilerState::LOOKING_FOR_OP1_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = CompilerState::LOOKING_FOR_OP1_END;
				instruction.operand1 = currentSymbol;
			}
		}
		break;
		case CompilerState::LOOKING_FOR_OP1_END: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = CompilerState::LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (currentSymbol == ',')
				state = CompilerState::LOOKING_FOR_OP2_START;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				state = CompilerState::LOOKING_FOR_SEPARATOR;
			else
				instruction.operand1 += currentSymbol;
		}
		break;
		case CompilerState::LOOKING_FOR_SEPARATOR: {
			if (currentSymbol == ',')
				state = CompilerState::LOOKING_FOR_OP2_START;
			else if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = CompilerState::LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == ']') || (currentSymbol == '(') || (currentSymbol == ')'))
				return false;
		}
		break;
		case CompilerState::LOOKING_FOR_COMPARATOR: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!'))
				instruction.comp += currentSymbol;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != '('))
				state = CompilerState::LOOKING_FOR_OP2_START;
			else {
				state = CompilerState::LOOKING_FOR_OP2_END;
				instruction.operand2 = currentSymbol;
			}
		}
		break;
		case CompilerState::LOOKING_FOR_OP2_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = CompilerState::LOOKING_FOR_OP2_END;
				instruction.operand2 = currentSymbol;
				if (bytePos == (codeSize - 1))
					instruction.readingFinished = true;
			}
		}
		break;
		case CompilerState::LOOKING_FOR_OP2_END: {
			if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				instruction.readingFinished = true;
			else {
				instruction.operand2 += currentSymbol;
				if ((bytePos + 1) == codeSize)
					instruction.readingFinished = true;
			}
		}
		break;
		}
		if ((currentSymbol == '\n') || ((bytePos + 1) == codeSize)) {
			if (!instruction.name.isEmpty()) {
				instruction.readingFinished = true;
			}
		}
		return true;
	}

	QString applyTableToCode(SymbolTable const* symbols, QString s)
	{
		QString prefix;
		QString postfix;
		for (int i = 0; i < 2; ++i) {
			if (s.left(1) == "[" || s.left(1) == "(") {
				prefix = prefix + s.left(1);
				s = s.mid(1);
			}
		}
		for (int i = 0; i < 2; ++i) {
			if (s.right(1) == "]" || s.right(1) == ")") {
				postfix = s.right(1) + postfix;
				s.chop(1);
			}
		}
		s = QString::fromStdString(symbols->getValue(s.toStdString()));
		return prefix + s + postfix;
	}

	bool resolveInstructionAndReturnSuccess(SymbolTable const* symbols, InstructionCoded& instructionCoded, InstructionUncoded instructionUncoded)
	{
		instructionUncoded.operand1 = applyTableToCode(symbols, instructionUncoded.operand1);
		instructionUncoded.operand2 = applyTableToCode(symbols, instructionUncoded.operand2);

		//prepare data for instruction coding
		if (instructionUncoded.name.toLower() == "mov")
			instructionCoded.operation = Enums::ComputerOperation::MOV;
		else if (instructionUncoded.name.toLower() == "add")
			instructionCoded.operation = Enums::ComputerOperation::ADD;
		else if (instructionUncoded.name.toLower() == "sub")
			instructionCoded.operation = Enums::ComputerOperation::SUB;
		else if (instructionUncoded.name.toLower() == "mul")
			instructionCoded.operation = Enums::ComputerOperation::MUL;
		else if (instructionUncoded.name.toLower() == "div")
			instructionCoded.operation = Enums::ComputerOperation::DIV;
		else if (instructionUncoded.name.toLower() == "xor")
			instructionCoded.operation = Enums::ComputerOperation::XOR;
		else if (instructionUncoded.name.toLower() == "or")
			instructionCoded.operation = Enums::ComputerOperation::OR;
		else if (instructionUncoded.name.toLower() == "and")
			instructionCoded.operation = Enums::ComputerOperation::AND;
		else if (instructionUncoded.name.toLower() == "if") {
			if (instructionUncoded.comp.toLower() == ">")
				instructionCoded.operation = Enums::ComputerOperation::IFG;
			else if ((instructionUncoded.comp.toLower() == ">=") || (instructionUncoded.comp.toLower() == "=>"))
				instructionCoded.operation = Enums::ComputerOperation::IFGE;
			else if ((instructionUncoded.comp.toLower() == "=") || (instructionUncoded.comp.toLower() == "=="))
				instructionCoded.operation = Enums::ComputerOperation::IFE;
			else if (instructionUncoded.comp.toLower() == "!=")
				instructionCoded.operation = Enums::ComputerOperation::IFNE;
			else if ((instructionUncoded.comp.toLower() == "<=") || (instructionUncoded.comp.toLower() == "=<"))
				instructionCoded.operation = Enums::ComputerOperation::IFLE;
			else if (instructionUncoded.comp.toLower() == "<")
				instructionCoded.operation = Enums::ComputerOperation::IFL;
			else {
				return false;
			}
		}
		else if (instructionUncoded.name.toLower() == "else")
			instructionCoded.operation = Enums::ComputerOperation::ELSE;
		else if (instructionUncoded.name.toLower() == "endif")
			instructionCoded.operation = Enums::ComputerOperation::ENDIF;
		else {
			return false;
		}

		if (instructionCoded.operation != Enums::ComputerOperation::ELSE && instructionCoded.operation != Enums::ComputerOperation::ENDIF) {
			if ((instructionUncoded.operand1.left(2) == "[[") && (instructionUncoded.operand1.right(2) == "]]")) {
				instructionCoded.opType1 = Enums::ComputerOptype::MEMMEM;
				instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 2);
				instructionUncoded.operand1.chop(2);
			}
			else if ((instructionUncoded.operand1.left(1) == "[") && (instructionUncoded.operand1.right(1) == "]")) {
				instructionCoded.opType1 = Enums::ComputerOptype::MEM;
				instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
				instructionUncoded.operand1.chop(1);
			}
			else if ((instructionUncoded.operand1.left(1) == "(") && (instructionUncoded.operand1.right(1) == ")")) {
				instructionCoded.opType1 = Enums::ComputerOptype::CMEM;
				instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
				instructionUncoded.operand1.chop(1);
			}
			else {
				return false;
			}

			if ((instructionUncoded.operand2.left(2) == "[[") && (instructionUncoded.operand2.right(2) == "]]")) {
				instructionCoded.opType2 = Enums::ComputerOptype::MEMMEM;
				instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 2);
				instructionUncoded.operand2.chop(2);
			}
			else if ((instructionUncoded.operand2.left(1) == "[") && (instructionUncoded.operand2.right(1) == "]")) {
				instructionCoded.opType2 = Enums::ComputerOptype::MEM;
				instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
				instructionUncoded.operand2.chop(1);
			}
			else if ((instructionUncoded.operand2.left(1) == "(") && (instructionUncoded.operand2.right(1) == ")")) {
				instructionCoded.opType2 = Enums::ComputerOptype::CMEM;
				instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
				instructionUncoded.operand2.chop(1);
			}
			else
				instructionCoded.opType2 = Enums::ComputerOptype::CONST;


			if (instructionUncoded.operand1.left(2) == "0x") {
				bool ok(true);
				instructionCoded.operand1 = instructionUncoded.operand1.remove(0, 2).toInt(&ok, 16);
				if (!ok) {
					return false;
				}
			}
			else {
				bool ok(true);
				instructionCoded.operand1 = instructionUncoded.operand1.toInt(&ok, 10);
				if (!ok)
					return false;
			}
			if (instructionUncoded.operand2.left(2) == "0x") {
				bool ok(true);
				instructionCoded.operand2 = instructionUncoded.operand2.remove(0, 2).toInt(&ok, 16);
				if (!ok) {
					return false;
				}
			}
			else {
				bool ok(true);
				instructionCoded.operand2 = instructionUncoded.operand2.toInt(&ok, 10);
				if (!ok) {
					return false;
				}
			}
		}
		else {
			instructionCoded.operand1 = 0;
			instructionCoded.operand2 = 0;
		}
		return true;
	}
}


CellComputerCompilerImpl::CellComputerCompilerImpl(QObject * parent) : CellComputerCompiler(parent)
{
	
}

void CellComputerCompilerImpl::init(SymbolTable const* symbols, SimulationParameters const& parameters)
{
	_symbols = symbols;
	_parameters = parameters;
}

CompilationResult CellComputerCompilerImpl::compileSourceCode(std::string const & code) const
{
	CompilerState state = CompilerState::LOOKING_FOR_INSTR_START;

	CompilationResult result;
	int linePos = 0;
	InstructionUncoded instructionUncoded;
	InstructionCoded instructionCoded;
	for (int bytePos = 0; bytePos < code.length(); ++bytePos) {
		QChar currentSymbol = code[bytePos];

		if (!gotoNextStateAndReturnSuccess(state, currentSymbol, instructionUncoded, bytePos, code.length())) {
			result.compilationOk = false;
			result.lineOfFirstError = linePos;
			return result;
		}
		if (instructionUncoded.readingFinished) {
			linePos++;
			if (!resolveInstructionAndReturnSuccess(_symbols, instructionCoded, instructionUncoded)) {
				result.compilationOk = false;
				result.lineOfFirstError = linePos;
				return result;
			}
			CompilerHelper::writeInstruction(result.compilation, instructionCoded);
			state = CompilerState::LOOKING_FOR_INSTR_START;
			instructionUncoded = InstructionUncoded();
		}
	}
	if (state == CompilerState::LOOKING_FOR_INSTR_START)
		result.compilationOk = true;
	else {
		result.compilationOk = false;
		result.lineOfFirstError = linePos;
	}
	return result;
}

std::string CellComputerCompilerImpl::decompileSourceCode(QByteArray const & data) const
{
	QString text;
	QString textOp1, textOp2;
	int conditionLevel = 0;
    auto const dataSize = (data.size() / 3) * 3;
	for (int instructionPointer = 0; instructionPointer < dataSize; ) {

		//decode instruction data
		InstructionCoded instruction;
		CompilerHelper::readInstruction(data, instructionPointer, instruction);

		//write spacing
		for (int j = 0; j < conditionLevel; ++j)
			text += "  ";

		//write operation
		if (instruction.operation == Enums::ComputerOperation::MOV)
			text += "mov";
		if (instruction.operation == Enums::ComputerOperation::ADD)
			text += "add";
		if (instruction.operation == Enums::ComputerOperation::SUB)
			text += "sub";
		if (instruction.operation == Enums::ComputerOperation::MUL)
			text += "mul";
		if (instruction.operation == Enums::ComputerOperation::DIV)
			text += "div";
		if (instruction.operation == Enums::ComputerOperation::XOR)
			text += "xor";
		if (instruction.operation == Enums::ComputerOperation::OR)
			text += "or";
		if (instruction.operation == Enums::ComputerOperation::AND)
			text += "and";
		if ((instruction.operation >= Enums::ComputerOperation::IFG) && (instruction.operation <= Enums::ComputerOperation::IFL)) {
			text += "if";
			++conditionLevel;
		}
		if (instruction.operation == Enums::ComputerOperation::ELSE) {
			if (conditionLevel > 0)
				text.chop(2);
			text += "else";
		}
		if (instruction.operation == Enums::ComputerOperation::ENDIF) {
			if (conditionLevel > 0) {
				text.chop(2);
				--conditionLevel;
			}
			text += "endif";
		}

		//write operands
		if (instruction.opType1 == Enums::ComputerOptype::MEM)
			textOp1 = "[" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand1, _parameters.tokenMemorySize), 0, 16) + "]";
		if (instruction.opType1 == Enums::ComputerOptype::MEMMEM)
			textOp1 = "[[" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand1, _parameters.tokenMemorySize), 0, 16) + "]]";
		if (instruction.opType1 == Enums::ComputerOptype::CMEM)
			textOp1 = "(" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand1, _parameters.cellFunctionComputerCellMemorySize), 0, 16) + ")";
		if (instruction.opType2 == Enums::ComputerOptype::MEM)
			textOp2 = "[" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand2, _parameters.tokenMemorySize), 0, 16) + "]";
		if (instruction.opType2 == Enums::ComputerOptype::MEMMEM)
			textOp2 = "[[" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand2, _parameters.tokenMemorySize), 0, 16) + "]]";
		if (instruction.opType2 == Enums::ComputerOptype::CMEM)
			textOp2 = "(" + QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand2, _parameters.cellFunctionComputerCellMemorySize), 0, 16) + ")";
		if (instruction.opType2 == Enums::ComputerOptype::CONST)
			textOp2 = QString("0x%1").arg(CompilerHelper::convertToAddress(instruction.operand2, _parameters.tokenMemorySize), 0, 16);

		//write separation/comparator
		if (instruction.operation <= Enums::ComputerOperation::AND) {
			text += " " + textOp1 + ", " + textOp2;
		}
		if (instruction.operation == Enums::ComputerOperation::IFG)
			text += " " + textOp1 + " > " + textOp2;
		if (instruction.operation == Enums::ComputerOperation::IFGE)
			text += " " + textOp1 + " >= " + textOp2;
		if (instruction.operation == Enums::ComputerOperation::IFE)
			text += " " + textOp1 + " = " + textOp2;
		if (instruction.operation == Enums::ComputerOperation::IFNE)
			text += " " + textOp1 + " != " + textOp2;
		if (instruction.operation == Enums::ComputerOperation::IFLE)
			text += " " + textOp1 + " <= " + textOp2;
		if (instruction.operation == Enums::ComputerOperation::IFL)
			text += " " + textOp1 + " < " + textOp2;
		if (instructionPointer < dataSize)
			text += "\n";
	}
	return text.toStdString();
}
