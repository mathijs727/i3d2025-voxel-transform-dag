#include "events.h"
#include "cuda_error_check.h"
#include "stats.h"
#include <chrono>
#include <iostream>

EventsManager::EventsManager(StatsRecorder* pStatsRecorder)
    : m_pStatsRecorder(pStatsRecorder)
{
}

EventsManager::~EventsManager()
{
    waitAndReportInFlightTimings();

    // Destroy events that were not queried yet.
    for (const auto& [_, timings] : m_namedTimings) {
        for (auto timing : timings) {
            cudaEventDestroy(timing.begin);
            cudaEventDestroy(timing.end);
        }
    }
    // Destroy fence updates that were not queried yet.
    for (const auto& [_, fence] : m_namedFences) {
        for (auto inFlightFence : fence.inFlightFences) {
            cudaEventDestroy(inFlightFence.ev);
        }
    }
    // Destroy events that were recycled.
    for (auto ev : m_recycledEvents)
        cudaEventDestroy(ev);
    // Destroy events used for synchronizing frames between CPU & GPU.
    for (auto ev : m_frameEvents)
        cudaEventDestroy(ev);

    CUDA_CHECK_ERROR();
}

AsyncTimingSection EventsManager::createTiming(const char* pName)
{
    // Create/recycle a begin and end event.
    // Record the begin event immediately.
    AsyncTimingSection out {};
    out.begin = getEvent();
    out.end = getEvent();
    out.pManager = this;
    out.name = pName;
    if (m_pStatsRecorder)
        out.timestamp = m_pStatsRecorder->get_frame_timestamp();
    cudaEventRecord(out.begin);
    CUDA_CHECK_ERROR();
    return out;
}

std::chrono::duration<double> EventsManager::getLastCompletedTiming(const char* pName)
{
    // Check if there are outstanding timings for this section name.
    const auto iter = m_namedTimings.find(std::string_view(pName));
    if (iter == std::end(m_namedTimings))
        return {};

    std::chrono::duration<double, std::milli> elapsed { 0 };
    auto& timings = iter->second;
    while (!timings.empty()) {
        // For each timing which the GPU has processed (FIFO order)
        const auto timing = timings.back();
        auto stat = cudaEventQuery(timing.end);
        if (stat != cudaSuccess)
            break;
        checkAlways(cudaEventQuery(timing.begin) == cudaSuccess);

        // Get timing from this event and let the begin/end events be reused in the future.
        float elapsedMs;
        cudaEventElapsedTime(&elapsedMs, timing.begin, timing.end);
        CUDA_CHECK_ERROR();
        elapsed = std::chrono::duration<double, std::milli>(elapsedMs);
        m_recycledEvents.push_front(timing.begin);
        m_recycledEvents.push_front(timing.end);

        if (m_pStatsRecorder)
            m_pStatsRecorder->reportAsync(timing.timestamp, pName, elapsed, Device::GPU);

        timings.pop_back();
    }
    return elapsed;
}

void EventsManager::waitAndReportInFlightTimings()
{
    // Wait for the GPU to finish all work.
    cudaDeviceSynchronize();
    // Iterate over all timings, call "getLastCompletedTiming" which internally will report to m_pStatsRecorder.
    for (const auto& [name, _] : m_namedTimings) {
        (void)getLastCompletedTiming(name.data());
    }
    // Verify that we have indeed reported all timings.
    for (const auto& [_, timings] : m_namedTimings) {
        checkAlways(timings.empty());
    }
}

void EventsManager::insertFenceValue(const char* pName, size_t fenceValue)
{
    auto ev = getEvent();
    cudaEventRecord(ev);
    m_namedFences[std::string_view(pName)].inFlightFences.push_front({ .ev = ev, .newFenceValue = fenceValue });
}

size_t EventsManager::getLastCompletedFenceValue(const char* pName)
{
    auto& fence = m_namedFences[std::string_view(pName)];
    // Update the fence value with any fenceEvent which the GPU has finished processing.
    while (!fence.inFlightFences.empty()) {
        // Loop over all finished events.
        const auto inFlightFence = fence.inFlightFences.back();
        if (cudaEventQuery(inFlightFence.ev) != cudaSuccess)
            break;

        // Update the fence value and recycle the event.
        fence.fenceValue = inFlightFence.newFenceValue;
        m_recycledEvents.push_front(inFlightFence.ev);
        fence.inFlightFences.pop_back();
    }

    return fence.fenceValue;
}

cudaEvent_t EventsManager::getEvent()
{
    if (!m_recycledEvents.empty()) {
        auto out = m_recycledEvents.back();
        m_recycledEvents.pop_back();
        return out;
    } else {
        cudaEvent_t out;
        cudaEventCreate(&out);
        CUDA_CHECK_ERROR();
        return out;
    }
}

void EventsManager::addSubmittedTiming(const AsyncTimingSection& timing)
{
    CUDA_CHECK_ERROR();
    cudaEventRecord(timing.end);
    CUDA_CHECK_ERROR();
    m_namedTimings[timing.name].push_front({ .begin = timing.begin, .end = timing.end, .timestamp = timing.timestamp });
}

AsyncTimingSection::AsyncTimingSection(AsyncTimingSection&& other)
    : begin(other.begin)
    , end(other.end)
    , pManager(other.pManager)
    , name(other.name)
{
    other.pManager = nullptr;
}

AsyncTimingSection::~AsyncTimingSection()
{
    if (pManager)
        pManager->addSubmittedTiming(*this);
}

void AsyncTimingSection::finish()
{
    pManager->addSubmittedTiming(*this);
    pManager = nullptr;
}
