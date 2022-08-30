//
//  ModelFile.h
//  cytocast
//
//  Created by Áron Takács on 2019. 05. 22..
//

#pragma once
#include <array>

#include "JSONDecoder.h"

namespace parser {
    struct Agent : public jsond::JSONDecodable<Agent> {};
}// namespace parser
