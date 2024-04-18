FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["neuraal_netwerk/neuraal_netwerk.csproj", "neuraal_netwerk/"]
RUN dotnet restore "neuraal_netwerk/neuraal_netwerk.csproj"
COPY . .
WORKDIR "/src/neuraal_netwerk"
RUN dotnet build "neuraal_netwerk.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "neuraal_netwerk.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "neuraal_netwerk.dll"]
