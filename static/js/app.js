const stateFilter = document.querySelector("#state_filter");
const countySelect = document.querySelector("#fips");

if (stateFilter && countySelect) {
    const options = Array.from(countySelect.options);

    stateFilter.addEventListener("change", () => {
        const selectedState = stateFilter.value;

        options.forEach((option) => {
            if (!option.value) {
                option.hidden = false;
                return;
            }
            option.hidden = selectedState && option.dataset.state !== selectedState;
        });

        const current = countySelect.selectedOptions[0];
        if (current && current.hidden) {
            countySelect.value = "";
        }
    });

    const selectedCounty = countySelect.selectedOptions[0];
    if (selectedCounty && selectedCounty.dataset.state) {
        stateFilter.value = selectedCounty.dataset.state;
        stateFilter.dispatchEvent(new Event("change"));
    }
}
