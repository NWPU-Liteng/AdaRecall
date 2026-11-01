#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "addr_mapper/addr_mapper.h"
#include "base/config.h"
#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "dram_controller/plugin.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class LPDDR5RD16Mapper final : public IAddrMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, LPDDR5RD16Mapper, "LPDDR5RD16", "Byte-addressed two-channel x16 RD16 mapping.");
 public:
  void init() override {}

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    auto dram = memory_system->get_ifce<IDRAM>();
    if (dram->m_impl->get_name() != "LPDDR5" || dram->m_channel_width != 16 ||
        dram->m_organization.dq != 16 ||
        dram->m_organization.count != std::vector<int>({2, 1, 4, 4, 65536, 1024})) {
      throw std::runtime_error("RD16 adapter requires two LPDDR5 16Gb x16 devices");
    }
  }

  void apply(Request& request) override {
    if (request.addr < 0 || request.addr % 32 || request.addr >= (int64_t(1) << 32)) {
      throw std::runtime_error("RD16 byte address is unaligned or exceeds 4GiB");
    }
    auto address = uint64_t(request.addr) >> 5;
    int channel = address & 1;
    address >>= 1;
    int column = int(address & 63) * 16;
    address >>= 6;
    int bankgroup = address & 3;
    address >>= 2;
    int bank = address & 3;
    address >>= 2;
    request.addr_vec = {channel, 0, bankgroup, bank, int(address), column};
  }
};

class MaterializationCommandAudit final : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, MaterializationCommandAudit, "MaterializationCommandAudit", "Per-channel issued-command accounting.");
 private:
  std::string path;
  IDRAM* dram = nullptr;
  std::map<std::string, uint64_t> counts;
 public:
  void init() override {
    path = param<std::string>("path").required();
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    m_ctrl = cast_parent<IDRAMController>();
    dram = m_ctrl->m_dram;
  }

  void update(bool request_found, ReqBuffer::iterator& request) override {
    if (request_found) {
      counts[std::string(dram->m_commands(request->command))]++;
    }
  }

  void finalize() override {
    YAML::Node report;
    report["channel"] = m_ctrl->m_channel_id;
    for (const auto& [command, count] : counts) {
      report["commands"][command] = count;
    }
    std::ofstream output(path + ".ch" + std::to_string(m_ctrl->m_channel_id) + ".yaml");
    output << report;
    if (!output) {
      throw std::runtime_error("cannot write channel audit");
    }
  }
};

}

int main(int argc, char** argv) {
  using namespace Ramulator;
  try {
    if (argc != 4) {
      throw std::runtime_error("usage: driver config.yaml input.trace report.yaml");
    }
    auto config = YAML::LoadFile(argv[1]);
    auto memory = Factory::create_memory_system(config);
    memory->connect_frontend(nullptr);
    auto dram = memory->get_ifce<IDRAM>();
    auto mapper = memory->get_ifce<IAddrMapper>();
    auto tail_cycles = std::max(dram->m_timing_vals("nCWL") + dram->m_timing_vals("nBL16") +
                                dram->m_timing_vals("nWR"), dram->m_timing_vals("nRFCab"));
    if (tail_cycles < 1 || config["MemorySystem"]["DRAM"]["impl"].as<std::string>() != "LPDDR5" ||
        config["MemorySystem"]["AddrMapper"]["impl"].as<std::string>() != "LPDDR5RD16" ||
        dram->m_channel_width != 16 || dram->m_organization.dq != 16 ||
        dram->m_organization.count != std::vector<int>({2, 1, 4, 4, 65536, 1024})) {
      throw std::runtime_error("missing RD16 mapping or write/recovery timing");
    }
    uint64_t cycle = 0, reads = 0, writes = 0, completed_reads = 0, rejected = 0;
    auto tick = [&]() {
      if (++cycle > 1000000000) {
        throw std::runtime_error("simulation exceeded fixed 1e9-cycle limit");
      }
      memory->tick();
    };
    YAML::Node report;
    for (int64_t address : std::vector<int64_t>({0, 32, 64, 4096, 16384, 65536, (int64_t(1) << 32) - 32})) {
      Request sample(address, Request::Type::Read);
      mapper->apply(sample);
      YAML::Node mapping;
      mapping["address"] = address;
      mapping["coordinates"] = sample.addr_vec;
      report["mapping_samples"].push_back(mapping);
    }
    std::map<std::string, uint64_t> bank_requests;
    std::ifstream input(argv[2]);
    if (!input) {
      throw std::runtime_error("cannot read trace");
    }
    std::string operation, operand;
    bool last_was_fence = false;
    uint64_t fences = 0;
    while (input >> operation) {
      if (!(input >> operand)) {
        throw std::runtime_error("incomplete trace record");
      }
      if (operation == "FENCE") {
        size_t parsed = 0;
        auto fence_id = std::stoull(operand, &parsed);
        if (parsed != operand.size() || fence_id != fences) {
          throw std::runtime_error("fence IDs must be dense and ordered");
        }
        int quiet_cycles = 0;
        while (quiet_cycles < tail_cycles || memory->is_pending()) {
          tick();
          quiet_cycles = memory->is_pending() ? 0 : quiet_cycles + 1;
        }
        if (completed_reads != reads) {
          throw std::runtime_error("drained read callback count mismatch");
        }
        YAML::Node fence;
        fence["phase"] = fences++;
        fence["cycle"] = cycle;
        fence["accepted_reads"] = reads;
        fence["accepted_writes"] = writes;
        fence["completed_reads"] = completed_reads;
        fence["pending"] = memory->is_pending();
        fence["quiet_tail_cycles"] = quiet_cycles;
        report["fences"].push_back(fence);
        last_was_fence = true;
        continue;
      }
      if (operation != "LD" && operation != "ST") {
        throw std::runtime_error("unsupported trace operation: " + operation);
      }
      size_t parsed = 0;
      auto address = std::stoll(operand, &parsed, 0);
      if (parsed != operand.size()) {
        throw std::runtime_error("malformed byte address");
      }
      Request request(address, operation == "LD" ? Request::Type::Read : Request::Type::Write,
                      0, [&](Request& finished) {
                        if (finished.type_id != Request::Type::Read) {
                          throw std::runtime_error("unexpected non-read completion callback");
                        }
                        completed_reads++;
                      });
      mapper->apply(request);
      auto bank_key = fmt::format("{}/{}/{}", request.addr_vec[0], request.addr_vec[2], request.addr_vec[3]);
      while (!memory->send(request)) {
        rejected++;
        tick();
      }
      bank_requests[bank_key]++;
      if (operation == "LD") {
        reads++;
      } else {
        writes++;
      }
      last_was_fence = false;
    }
    if (!input.eof() || !last_was_fence || !fences) {
      throw std::runtime_error("trace must end with a complete FENCE record");
    }
    report["complete"] = true;
    report["cycles"] = cycle;
    report["tck_ns"] = memory->get_tCK();
    report["read_requests"] = reads;
    report["write_requests"] = writes;
    report["completed_reads"] = completed_reads;
    report["backpressure_rejections"] = rejected;
    report["write_refresh_tail_cycles"] = tail_cycles;
    report["pending_at_finish"] = memory->is_pending();
    for (const auto& [bank, count] : bank_requests) {
      report["bank_requests"][bank] = count;
    }
    memory->finalize();
    std::ofstream output(argv[3]);
    output << report;
    if (!output) {
      throw std::runtime_error("cannot write driver report");
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << std::endl;
    return 1;
  }
}
