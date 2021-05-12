#include "JSONWriter.h"
#include "rapidjson/stringbuffer.h"
#include <rapidjson/writer.h>
#include "rapidjson/prettywriter.h"
#include <fstream>

JSONWriter::JSONWriter() : allocator(d.GetAllocator()) { d.SetObject(); }

rapidjson::Value JSONWriter::stringToObject(const std::string& in) {
    rapidjson::Value obj(rapidjson::kObjectType);
    obj.SetString(in.c_str(), in.length(), allocator);
    return obj;
}

void JSONWriter::writeFile(const std::string& fileName) const {
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    // writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
    writer.SetMaxDecimalPlaces(4);
    d.Accept(writer);
    std::ofstream outFile;
    outFile.open(fileName.c_str());
    outFile << buffer.GetString();
    outFile.close();
}