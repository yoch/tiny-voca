# Run from work/abl: regions of the reference (abl/ref), the kit (abl/kit) and the final (abl/fin) sources.
# Extract code regions from a source by start/end markers.
def region(s, start, end):
    i = s.index(start); j = s.index(end, i)
    return s[i:j]
ref = open('ref/src/source_filter.cpp').read()
kit = open('kit/src/source_filter.cpp').read()
# pulse body: from "    double total = 0.0;" to "    return out;\n}\n\n// Time-varying"
P0 = "    std::vector<float> out(static_cast<size_t>(frames * hop));\n"
P1 = "    return out;\n}\n\n// Time-varying"
pulse_ref, pulse_kit = region(ref, P0, P1), region(kit, P0, P1)
H0 = "    std::vector<float> head(static_cast<size_t>(frames * outputs));\n"
H1 = "    // The filter frames are worked"
head_ref, head_kit = region(ref, H0, H1), region(kit, H0, H1)
C0 = "    for (const Branch * branch : {&harmonic_, &noise_}) {"
C0k = "    for (Branch * branch : {&harmonic_, &noise_}) {"
C1 = "\n}\n\nstd::vector<float> TinyVocaSourceFilter::pulse_train"
ctor_ref, ctor_kit = region(ref, C0, C1), region(kit, C0k, C1)
