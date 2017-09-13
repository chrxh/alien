#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

class NumberGeneratorTest : public ::testing::Test
{
public:
	NumberGeneratorTest();
	~NumberGeneratorTest();

protected:
	NumberGenerator* _numberGen = nullptr;
};

NumberGeneratorTest::NumberGeneratorTest()
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_numberGen = factory->buildRandomNumberGenerator();
}

NumberGeneratorTest::~NumberGeneratorTest()
{
	delete _numberGen;
}


TEST_F(NumberGeneratorTest, testTags)
{
	_numberGen->init(123, 1);
	quint64 tag = _numberGen->getTag();
	EXPECT_EQ(1, tag >> 48);
	EXPECT_EQ(1, tag & 0xffffffffffff);
	tag = _numberGen->getTag();
	EXPECT_EQ(1, tag >> 48);
	EXPECT_EQ(2, tag & 0xffffffffffff);
	tag = _numberGen->getTag();
	EXPECT_EQ(1, tag >> 48);
	EXPECT_EQ(3, tag & 0xffffffffffff);

	_numberGen->init(123, 23);
	tag = _numberGen->getTag();
	EXPECT_EQ(23, tag >> 48);
	EXPECT_EQ(1, tag & 0xffffffffffff);
	tag = _numberGen->getTag();
	EXPECT_EQ(23, tag >> 48);
	EXPECT_EQ(2, tag & 0xffffffffffff);
}

