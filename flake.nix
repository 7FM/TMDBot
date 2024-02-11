{
  description = "TMDBot env flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils}:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
    in {
      devShell = with pkgs; let
        myPyPackages = python-packages: with python-packages; [
          requests
          python-telegram-bot
          autopep8 # autoformatter
          pyyaml
          (buildPythonPackage rec {
            pname = "tmdbv3api";
            version = "1.9.0";

            src = fetchPypi {
              inherit pname version;
              sha256 = "sha256-UExdprmcRRb/FgoBV2ES0JfyCcBTT5Q8FcS1bL2Swzs=";
            };

            propagatedBuildInputs = [
              requests
              urllib3
            ];

            doCheck = false;

            pythonImportsCheck = [ "tmdbv3api" ];
          })
        ];

        myPythonWithPackages = pkgs.python3.withPackages myPyPackages;

      # TODO use stdEnv?
      in mkShellNoCC {
        nativeBuildInputs = [
          # Python env for the utility scripts & cocotb
          myPythonWithPackages
        ];
        hardeningDisable = [ "all" ];
      };
    });}
